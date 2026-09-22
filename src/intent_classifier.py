"""AI Intent Classifier: Semantic Embedding & Hybrid Classifier with structured explainability."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.schemas import IntentClassificationResult
from src.preprocessing import clean_text
from src.intent_taxonomy import INTENT_TAXONOMY, get_all_intent_ids, IntentSpec

logger = logging.getLogger(__name__)

HOW_TO_PATTERN = re.compile(r"\b(how (do|can|to|should|would)|where (do|can)|can i|is it possible|steps to|guide)\b", re.IGNORECASE)
BUG_CRASH_PATTERN = re.compile(r"\b(bug|glitch|crash|crashes|freeze|freezes|freezing|stuck|lag|lagging|broken|reboot|loop)\b", re.IGNORECASE)
FINANCIAL_PATTERN = re.compile(r"\b(refund|charged?|bill|billing|subscription|credit card|unauthorized|itunes\.com/bill|payment)\b", re.IGNORECASE)
SECURITY_PATTERN = re.compile(r"\b(locked?|password|apple id|icloud|2fa|two-factor|verification code|passcode|hacked?)\b", re.IGNORECASE)
BATTERY_PATTERN = re.compile(r"\b(battery|drain|draining|charge|charging|overheat(ing)?|die|dying|percent(age)?)\b", re.IGNORECASE)
HARDWARE_PATTERN = re.compile(r"\b(screen|cracked?|display|speaker|audio|mic|microphone|airpods?|earpiece|static|camera)\b", re.IGNORECASE)
COMPLAINT_PATTERN = re.compile(r"\b(terrible|worst|horrible|useless|unacceptable|supervisor|manager|human agent|lawyer|sue|switch(ing)? to (samsung|android))\b", re.IGNORECASE)
SHIPPING_PATTERN = re.compile(r"\b(order|shipping|delivery|track(ing)?|trade-in|ups|fedex|package|shipment|store pickup)\b", re.IGNORECASE)


class SemanticEmbeddingClassifier:
    """Semantic embedding classifier using intent prototypes, phrase matching, and confidence calibration."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.intent_ids: List[str] = get_all_intent_ids()
        self.intent_embeddings: Optional[np.ndarray] = None
        self._build_intent_prototypes()

    def _get_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer in SemanticEmbeddingClassifier ({e})")
                self.model = None
        return self.model

    def _build_intent_prototypes(self):
        """Builds multi-anchor semantic prototype vector for each intent based on definition, keywords, and examples."""
        model = self._get_model()
        if model is None:
            return

        proto_vectors = []
        for intent_id in self.intent_ids:
            spec: IntentSpec = INTENT_TAXONOMY[intent_id]
            # Create distinct anchor texts for balanced prototype embedding
            anchors = [
                f"{spec.name}: {spec.description}",
                f"Customer support inquiries about {', '.join(spec.keywords[:8])}",
                *spec.representative_examples
            ]
            anchor_embs = model.encode(anchors, normalize_embeddings=True)
            mean_emb = np.mean(anchor_embs, axis=0)
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            proto_vectors.append(mean_emb)

        self.intent_embeddings = np.array(proto_vectors, dtype=np.float32)

    def predict(self, text: str) -> IntentClassificationResult:
        clean_q = clean_text(text)
        if not clean_q:
            return IntentClassificationResult(
                intent="other",
                confidence=0.5,
                reason="Empty input text assigned to 'other'.",
                scores={i: 0.0 for i in self.intent_ids},
                top_alternatives=[],
                confidence_margin=0.0,
                semantic_similarity=0.0,
                keyword_evidence=[],
                is_uncertain=True
            )

        model = self._get_model()
        if model is None or self.intent_embeddings is None:
            return self._keyword_heuristic_predict(clean_q)

        q_emb = model.encode([clean_q], normalize_embeddings=True)
        sims = np.dot(self.intent_embeddings, q_emb.T).flatten()

        boosted_sims = sims.copy()
        lower_text = clean_q.lower()
        keyword_evidence: List[str] = []

        # Disambiguation & Phrase-level lexical gating
        is_how_to = bool(HOW_TO_PATTERN.search(clean_q))
        has_bug = bool(BUG_CRASH_PATTERN.search(clean_q))
        has_finance = bool(FINANCIAL_PATTERN.search(clean_q))
        has_security = bool(SECURITY_PATTERN.search(clean_q))
        has_battery = bool(BATTERY_PATTERN.search(clean_q))
        has_hardware = bool(HARDWARE_PATTERN.search(clean_q))
        has_complaint = bool(COMPLAINT_PATTERN.search(clean_q))
        has_shipping = bool(SHIPPING_PATTERN.search(clean_q))

        for idx, intent_id in enumerate(self.intent_ids):
            spec = INTENT_TAXONOMY[intent_id]
            matched_kws = [kw for kw in spec.keywords if kw.lower() in lower_text]
            if matched_kws:
                boosted_sims[idx] += min(0.12, len(matched_kws) * 0.04)
                keyword_evidence.extend(matched_kws)

        # Apply domain disambiguation logic
        # 1. If query is a how-to question without crash/bug indicators, boost general_how_to_and_features
        how_to_idx = self.intent_ids.index("general_how_to_and_features")
        bug_idx = self.intent_ids.index("ios_software_update_bugs")
        
        if is_how_to and not has_bug:
            boosted_sims[how_to_idx] += 0.18
            boosted_sims[bug_idx] -= 0.10

        if has_finance:
            fin_idx = self.intent_ids.index("app_store_billing_and_subscriptions")
            boosted_sims[fin_idx] += 0.20

        if has_security:
            sec_idx = self.intent_ids.index("apple_id_and_icloud")
            boosted_sims[sec_idx] += 0.20

        if has_battery and not (is_how_to and not has_bug):
            bat_idx = self.intent_ids.index("battery_and_charging")
            boosted_sims[bat_idx] += 0.15

        if has_hardware and not has_battery:
            hw_idx = self.intent_ids.index("hardware_audio_and_screen")
            boosted_sims[hw_idx] += 0.15

        if has_complaint:
            comp_idx = self.intent_ids.index("general_complaint_and_escalation")
            boosted_sims[comp_idx] += 0.20

        if has_shipping:
            ship_idx = self.intent_ids.index("order_shipping_and_trade_in")
            boosted_sims[ship_idx] += 0.20

        # Calibrated Softmax
        exp_sims = np.exp(boosted_sims * 4.5)
        probs = exp_sims / np.sum(exp_sims)

        sorted_indices = np.argsort(probs)[::-1]
        top_idx = int(sorted_indices[0])
        second_idx = int(sorted_indices[1])

        pred_intent = self.intent_ids[top_idx]
        confidence = float(probs[top_idx])
        second_intent = self.intent_ids[second_idx]
        second_conf = float(probs[second_idx])
        margin = float(confidence - second_conf)

        is_uncertain = (confidence < 0.45) or (margin < 0.08)

        top_alts = [(self.intent_ids[i], round(float(probs[i]), 4)) for i in sorted_indices[1:4]]
        score_dict = {i: round(float(p), 4) for i, p in zip(self.intent_ids, probs)}

        reason = (
            f"Classified as '{INTENT_TAXONOMY[pred_intent].name}' with confidence {confidence:.2f} "
            f"(margin: {margin:.2f} over '{second_intent}'). "
            f"Raw semantic similarity: {sims[top_idx]:.2f}."
        )

        return IntentClassificationResult(
            intent=pred_intent,
            confidence=round(confidence, 4),
            reason=reason,
            scores=score_dict,
            top_alternatives=top_alts,
            confidence_margin=round(margin, 4),
            semantic_similarity=round(float(sims[top_idx]), 4),
            keyword_evidence=list(set(keyword_evidence[:5])),
            is_uncertain=is_uncertain
        )

    def _keyword_heuristic_predict(self, text: str) -> IntentClassificationResult:
        lower = text.lower()
        scores = {}
        matched_kws_map = {}
        for intent_id, spec in INTENT_TAXONOMY.items():
            matches = [kw for kw in spec.keywords if kw.lower() in lower]
            scores[intent_id] = len(matches)
            matched_kws_map[intent_id] = matches

        total_matches = sum(scores.values())
        if total_matches == 0:
            return IntentClassificationResult(
                intent="other",
                confidence=0.40,
                reason="No intent keywords detected, defaulted to 'other'.",
                scores={i: round(1.0 / len(self.intent_ids), 4) for i in self.intent_ids},
                top_alternatives=[],
                confidence_margin=0.0,
                semantic_similarity=0.0,
                keyword_evidence=[],
                is_uncertain=True
            )

        sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_intent, best_count = sorted_intents[0]
        second_intent, second_count = sorted_intents[1]

        conf = min(0.95, 0.45 + (best_count / total_matches) * 0.50)
        margin = (best_count - second_count) / total_matches

        return IntentClassificationResult(
            intent=best_intent,
            confidence=round(conf, 4),
            reason=f"Matched {best_count} keyword(s) {matched_kws_map[best_intent][:3]} for '{INTENT_TAXONOMY[best_intent].name}'.",
            scores={k: round(v / total_matches, 4) for k, v in scores.items()},
            top_alternatives=[(k, round(v / total_matches, 4)) for k, v in sorted_intents[1:4]],
            confidence_margin=round(margin, 4),
            semantic_similarity=0.0,
            keyword_evidence=matched_kws_map[best_intent][:5],
            is_uncertain=(conf < 0.50 or margin < 0.10)
        )


class HybridIntentClassifier:
    """Ensemble combining TF-IDF model, Semantic Embeddings, and Domain Rule heuristics."""

    def __init__(self, tfidf_classifier=None, embedding_classifier: Optional[SemanticEmbeddingClassifier] = None):
        self.tfidf_classifier = tfidf_classifier
        self.embedding_classifier = embedding_classifier or SemanticEmbeddingClassifier()

    def predict(self, text: str) -> IntentClassificationResult:
        emb_res = self.embedding_classifier.predict(text)
        
        if self.tfidf_classifier is not None and getattr(self.tfidf_classifier, "pipeline", None) is not None:
            try:
                tfidf_res = self.tfidf_classifier.predict(text)
                combined_scores = {}
                all_intents = set(emb_res.scores.keys()).union(set(tfidf_res.scores.keys()))
                for i in all_intents:
                    s_emb = emb_res.scores.get(i, 0.0)
                    s_tfidf = tfidf_res.scores.get(i, 0.0)
                    # Weighted combination: 65% semantic embedding + 35% TF-IDF
                    combined_scores[i] = 0.65 * s_emb + 0.35 * s_tfidf

                sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                best_intent, conf = sorted_items[0]
                second_intent, second_conf = sorted_items[1]
                margin = conf - second_conf
                is_uncertain = (conf < 0.45) or (margin < 0.08)

                reason = (
                    f"Hybrid ensemble: Semantic embedding ({emb_res.intent}={emb_res.confidence:.2f}) "
                    f"+ TF-IDF ({tfidf_res.intent}={tfidf_res.confidence:.2f}) -> {best_intent} ({conf:.2f}, margin: {margin:.2f})."
                )
                return IntentClassificationResult(
                    intent=best_intent,
                    confidence=round(min(1.0, conf), 4),
                    reason=reason,
                    scores={k: round(v, 4) for k, v in combined_scores.items()},
                    top_alternatives=[(k, round(v, 4)) for k, v in sorted_items[1:4]],
                    confidence_margin=round(margin, 4),
                    semantic_similarity=emb_res.semantic_similarity,
                    keyword_evidence=emb_res.keyword_evidence,
                    is_uncertain=is_uncertain
                )
            except Exception as e:
                logger.warning(f"Error in TF-IDF ensemble step ({e}), falling back to embedding classifier.")

        return emb_res
