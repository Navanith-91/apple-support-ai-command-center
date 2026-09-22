"""Main Customer Support AI Agent pipeline combining preprocessing, classification, retrieval, generation, and escalation."""

import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from src.schemas import (
    AgentResponse, 
    IntentClassificationResult, 
    RetrievedEvidence, 
    GenerationResult, 
    EscalationResult,
    DecisionEnum
)
from src.preprocessing import clean_text, is_usable_message
from src.config import AppConfig, load_config
from src.intent_classifier import HybridIntentClassifier, SemanticEmbeddingClassifier
from src.retriever import HistoricalRetriever
from src.response_generator import GroundedResponseGenerator
from src.escalation import EscalationPolicyEngine
from evaluation.baselines import TfidfIntentClassifier

logger = logging.getLogger(__name__)


class CustomerSupportAgent:
    """Full end-to-end AI Customer Support Agent for the selected brand."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        retriever: Optional[HistoricalRetriever] = None,
        classifier: Optional[Any] = None,
        generator: Optional[GroundedResponseGenerator] = None,
        escalation_engine: Optional[EscalationPolicyEngine] = None
    ):
        self.config = config or load_config()
        self.brand = self.config.brand.selected

        # Initialize Retriever
        if retriever is not None:
            self.retriever = retriever
        else:
            index_path = self.config.data.resolve_path(self.config.data.index_path)
            if index_path.exists():
                self.retriever = HistoricalRetriever.load(index_path)
            else:
                logger.warning(f"Index file not found at {index_path}. Creating unindexed retriever.")
                self.retriever = HistoricalRetriever(
                    model_name=self.config.retrieval.model_name,
                    top_k=self.config.retrieval.top_k,
                    similarity_threshold=self.config.retrieval.similarity_threshold
                )

        # Initialize Intent Classifier
        if classifier is not None:
            self.classifier = classifier
        else:
            tfidf_path = self.config.data.resolve_path(self.config.data.tfidf_model_path)
            tfidf_clf = None
            if tfidf_path.exists():
                try:
                    tfidf_clf = TfidfIntentClassifier.load(tfidf_path)
                except Exception as e:
                    logger.warning(f"Could not load TF-IDF model from {tfidf_path}: {e}")

            emb_clf = SemanticEmbeddingClassifier(model_name=self.config.retrieval.model_name)
            self.classifier = HybridIntentClassifier(tfidf_classifier=tfidf_clf, embedding_classifier=emb_clf)

        # Initialize Response Generator
        if generator is not None:
            self.generator = generator
        else:
            self.generator = GroundedResponseGenerator(
                brand=self.brand,
                provider=self.config.generation.provider,
                model_name=self.config.generation.model,
                temperature=self.config.generation.temperature
            )

        # Initialize Escalation Engine
        if escalation_engine is not None:
            self.escalation_engine = escalation_engine
        else:
            self.escalation_engine = EscalationPolicyEngine(config=self.config.escalation)

    def run(self, customer_message: str, context: Optional[str] = "") -> AgentResponse:
        """Executes the full agent customer support workflow.
        
        Steps:
        1. Preprocessing and input validation
        2. Intent classification with confidence scoring & explainability
        3. Semantic historical retrieval with grounding checks
        4. Grounded reply generation with anti-hallucination guardrails
        5. Escalation policy decision (Safety-First)
        6. Structured response synthesis with full execution metadata
        """
        start_time = time.perf_counter()
        
        # Step 1: Preprocessing
        clean_msg = clean_text(customer_message)
        if not is_usable_message(clean_msg):
            return AgentResponse(
                intent="other",
                confidence=0.0,
                reply="Thanks for reaching out to Apple Support. Could you please provide more details on how we can help?",
                decision=DecisionEnum.AUTO_HANDLE.value,
                reason="Empty or minimal input message; returned clarification prompt.",
                evidence=[],
                metadata={
                    "total_latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                    "grounded": False,
                    "risk_signals": []
                }
            )

        # Step 2: Intent Classification
        t0 = time.perf_counter()
        classification: IntentClassificationResult = self.classifier.predict(clean_msg)
        t_clf = (time.perf_counter() - t0) * 1000

        # Step 3: Historical Retrieval
        t0 = time.perf_counter()
        evidence: List[RetrievedEvidence] = self.retriever.retrieve(
            query=clean_msg,
            top_k=self.config.retrieval.top_k,
            min_similarity=self.config.retrieval.similarity_threshold
        )
        t_ret = (time.perf_counter() - t0) * 1000

        # Step 4: Grounded Response Generation
        t0 = time.perf_counter()
        gen_result: GenerationResult = self.generator.generate_response(
            customer_message=clean_msg,
            intent=classification.intent,
            evidence=evidence
        )
        t_gen = (time.perf_counter() - t0) * 1000

        # Step 5: Escalation Policy Decision
        escalation: EscalationResult = self.escalation_engine.evaluate(
            customer_message=customer_message,
            classification=classification,
            evidence=evidence
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        # Format evidence for final output
        evidence_dicts = [ev.model_dump() for ev in evidence]

        return AgentResponse(
            intent=classification.intent,
            confidence=classification.confidence,
            reply=gen_result.reply,
            decision=escalation.decision.value,
            reason=escalation.reason,
            evidence=evidence_dicts,
            metadata={
                "total_latency_ms": round(total_latency_ms, 2),
                "classification_latency_ms": round(t_clf, 2),
                "retrieval_latency_ms": round(t_ret, 2),
                "generation_latency_ms": round(t_gen, 2),
                "risk_signals": escalation.risk_signals,
                "confidence_margin": getattr(classification, "confidence_margin", 0.0),
                "semantic_similarity": getattr(classification, "semantic_similarity", 0.0),
                "keyword_evidence": getattr(classification, "keyword_evidence", []),
                "grounded": getattr(gen_result, "grounded", True),
                "safety_notes": getattr(gen_result, "safety_notes", []),
                "classifier_reason": classification.reason
            }
        )
