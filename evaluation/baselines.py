"""Baseline models: Majority-class baseline, TF-IDF + Logistic Regression, and Keyword/Rule-Based baseline."""

import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from src.schemas import IntentClassificationResult
from src.preprocessing import clean_text
from src.intent_taxonomy import INTENT_TAXONOMY, get_all_intent_ids

logger = logging.getLogger(__name__)


class MajorityClassBaseline:
    """Baseline 1: Trivial baseline predicting the most common intent in the training set."""

    def __init__(self):
        self.majority_intent: Optional[str] = None
        self.prevalence: float = 0.0
        self.class_counts: Dict[str, int] = {}

    def fit(self, intents: List[str]) -> "MajorityClassBaseline":
        counts = Counter(intents)
        if not counts:
            raise ValueError("Cannot fit MajorityClassBaseline on empty intent list")
        self.majority_intent, top_count = counts.most_common(1)[0]
        self.prevalence = top_count / len(intents)
        self.class_counts = dict(counts)
        logger.info(f"MajorityClassBaseline fitted: '{self.majority_intent}' (prevalence: {self.prevalence:.2%})")
        return self

    def predict(self, text: str) -> IntentClassificationResult:
        if self.majority_intent is None:
            raise ValueError("Baseline is not fitted")
        return IntentClassificationResult(
            intent=self.majority_intent,
            confidence=round(self.prevalence, 4),
            reason=f"Predicted most frequent historical intent ({self.majority_intent}) with prevalence {self.prevalence:.2%}.",
            scores={self.majority_intent: round(self.prevalence, 4)}
        )

    def predict_batch(self, texts: List[str]) -> List[IntentClassificationResult]:
        return [self.predict(t) for t in texts]


class TfidfIntentClassifier:
    """Baseline 2: TF-IDF feature extraction + Calibrated Logistic Regression."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: int = 10000,
        C: float = 1.0,
        random_state: int = 42
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.C = C
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, texts: List[str], labels: List[str]) -> "TfidfIntentClassifier":
        cleaned_texts = [clean_text(t) for t in texts]
        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=True,
            stop_words="english"
        )
        base_clf = LogisticRegression(
            C=self.C,
            max_iter=1000,
            class_weight="balanced",
            random_state=self.random_state
        )
        min_samples = min(Counter(labels).values()) if labels else 0
        if min_samples >= 3:
            clf = CalibratedClassifierCV(estimator=base_clf, cv=3)
        else:
            clf = base_clf

        self.pipeline = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf)
        ])
        self.pipeline.fit(cleaned_texts, labels)
        self.classes_ = self.pipeline.classes_
        logger.info(f"TfidfIntentClassifier trained on {len(texts):,} samples across {len(self.classes_)} classes.")
        return self

    def predict(self, text: str) -> IntentClassificationResult:
        if self.pipeline is None or self.classes_ is None:
            raise ValueError("Classifier is not fitted")
        
        clean = clean_text(text)
        if not clean:
            # Handle empty or purely stripped messages
            return IntentClassificationResult(
                intent="other",
                confidence=0.5,
                reason="Empty or non-text message defaulted to 'other'.",
                scores={c: 0.0 for c in self.classes_}
            )

        probs = self.pipeline.predict_proba([clean])[0]
        max_idx = int(np.argmax(probs))
        pred_intent = str(self.classes_[max_idx])
        confidence = float(probs[max_idx])

        # Generate top feature explanation
        score_dict = {str(c): round(float(p), 4) for c, p in zip(self.classes_, probs)}
        
        reason = f"TF-IDF model classified as '{pred_intent}' with confidence {confidence:.2f}."
        return IntentClassificationResult(
            intent=pred_intent,
            confidence=round(confidence, 4),
            reason=reason,
            scores=score_dict
        )

    def predict_batch(self, texts: List[str]) -> List[IntentClassificationResult]:
        return [self.predict(t) for t in texts]

    def save(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "classes": self.classes_}, f)
        logger.info(f"Saved TF-IDF model to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "TfidfIntentClassifier":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found at {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        instance = cls()
        instance.pipeline = data["pipeline"]
        instance.classes_ = data["classes"]
        return instance


class RuleBasedBaseline:
    """Baseline 3: Transparent keyword/regex domain rule baseline."""

    def __init__(self):
        self.intent_ids = get_all_intent_ids()
        self.risk_keywords = [
            "refund", "charge", "charged", "unauthorized", "stolen", "hacked",
            "lawyer", "sue", "fraud", "dispute", "manager", "supervisor", "lock", "locked"
        ]

    def predict(self, text: str) -> IntentClassificationResult:
        clean = clean_text(text).lower()
        if not clean:
            return IntentClassificationResult(
                intent="other",
                confidence=0.5,
                reason="Empty input text assigned to 'other'.",
                scores={i: 0.0 for i in self.intent_ids}
            )

        match_counts = {}
        matched_keywords_per_intent = {}

        for intent_id in self.intent_ids:
            spec = INTENT_TAXONOMY[intent_id]
            matched = [kw for kw in spec.keywords if kw.lower() in clean]
            match_counts[intent_id] = len(matched)
            matched_keywords_per_intent[intent_id] = matched

        total_matches = sum(match_counts.values())

        if total_matches == 0:
            return IntentClassificationResult(
                intent="other",
                confidence=0.40,
                reason="No intent keywords matched; defaulted to 'other'.",
                scores={i: round(1.0 / len(self.intent_ids), 4) for i in self.intent_ids}
            )

        best_intent = max(match_counts, key=match_counts.get)
        best_count = match_counts[best_intent]
        # Confidence scaled by match proportion
        confidence = min(0.95, 0.45 + (best_count / total_matches) * 0.50)

        scores = {
            i: round(match_counts[i] / total_matches, 4) if total_matches > 0 else 0.0
            for i in self.intent_ids
        }

        matched_list = matched_keywords_per_intent[best_intent][:3]
        reason = f"Rule-based match: found keyword(s) {matched_list} for '{INTENT_TAXONOMY[best_intent].name}'."

        return IntentClassificationResult(
            intent=best_intent,
            confidence=round(confidence, 4),
            reason=reason,
            scores=scores
        )

    def predict_batch(self, texts: List[str]) -> List[IntentClassificationResult]:
        return [self.predict(t) for t in texts]

    def predict_decision(self, text: str, classification: IntentClassificationResult) -> str:
        """Determines escalation decision based on simple transparent rules."""
        text_lower = text.lower()
        # Check risk keywords
        for rk in self.risk_keywords:
            if rk in text_lower:
                return "ESCALATE_TO_HUMAN"

        # Check sensitive intents
        if classification.intent in ("app_store_billing_and_subscriptions", "apple_id_and_icloud", "general_complaint_and_escalation"):
            return "ESCALATE_TO_HUMAN"

        if classification.intent == "other" or classification.confidence < 0.50:
            return "ESCALATE_TO_HUMAN"

        return "AUTO_HANDLE"
