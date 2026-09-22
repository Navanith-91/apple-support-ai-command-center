"""Escalation and Risk Policy Engine.

Balances customer safety and automated efficiency:
1. HIGH RISK (Security, Billing disputes, Anger, Legal) -> ESCALATE_TO_HUMAN
2. LOW RISK + HIGH CONFIDENCE + STRONG HISTORICAL EVIDENCE -> AUTO_HANDLE
3. LOW CONFIDENCE / WEAK EVIDENCE / AMBIGUITY -> ESCALATE_TO_HUMAN
"""

import re
import logging
from typing import List, Optional

from src.schemas import DecisionEnum, EscalationResult, IntentClassificationResult, RetrievedEvidence
from src.config import EscalationConfig

logger = logging.getLogger(__name__)

# Explicit escalation and risk patterns
HUMAN_AGENT_REQUEST_PATTERN = re.compile(
    r"\b(speak to (a )?human|talk to (a )?person|real (agent|person)|manager|supervisor|representative|genius bar appointment)\b",
    re.IGNORECASE
)
LEGAL_ANGER_PATTERN = re.compile(
    r"\b(lawyer|attorney|sue|suing|lawsuit|fraud|scam|stolen|hacked|unacceptable|useless|worst service)\b",
    re.IGNORECASE
)
FINANCIAL_RISK_PATTERN = re.compile(
    r"\b(refund|charged? (twice|wrongly|without)|unauthorized charge|dispute charge|stolen card|bank statement)\b",
    re.IGNORECASE
)
SECURITY_RISK_PATTERN = re.compile(
    r"\b(locked account|apple id locked|forgot password|2fa loop|recovery key|account compromised|cant verify)\b",
    re.IGNORECASE
)


class EscalationPolicyEngine:
    """Evaluates risk signals, intent confidence, confidence margin, and retrieval grounding to decide escalation."""

    def __init__(self, config: Optional[EscalationConfig] = None):
        if config is None:
            self.min_confidence = 0.50
            self.min_retrieval_similarity = 0.40
            self.min_margin = 0.06
            self.auto_escalate_intents = [
                "general_complaint_and_escalation"
            ]
            self.risk_keywords = [
                "refund", "unauthorized", "stolen", "hacked", "lawyer", "sue",
                "fraud", "dispute", "manager", "supervisor"
            ]
        else:
            self.min_confidence = config.min_confidence
            self.min_retrieval_similarity = config.min_retrieval_similarity
            self.min_margin = 0.06
            self.auto_escalate_intents = config.auto_escalate_intents or ["general_complaint_and_escalation"]
            self.risk_keywords = config.risk_keywords or [
                "refund", "unauthorized", "stolen", "hacked", "lawyer", "sue", "fraud", "dispute"
            ]

    def evaluate(
        self,
        customer_message: str,
        classification: IntentClassificationResult,
        evidence: List[RetrievedEvidence]
    ) -> EscalationResult:
        """Determines whether a conversation can be AUTO_HANDLE or must ESCALATE_TO_HUMAN."""
        risk_signals: List[str] = []
        reasons: List[str] = []
        text_lower = customer_message.lower()

        top_similarity = float(evidence[0].score) if evidence else 0.0
        confidence = classification.confidence
        margin = getattr(classification, "confidence_margin", 0.0) or 0.0

        # Risk Rule 1: Explicit Human / Supervisor / Representative Demand
        if HUMAN_AGENT_REQUEST_PATTERN.search(customer_message):
            risk_signals.append("explicit_human_agent_request")
            reasons.append("Customer explicitly requested to speak with a human representative, manager, or supervisor.")

        # Risk Rule 2: Legal threats or severe anger
        if LEGAL_ANGER_PATTERN.search(customer_message):
            risk_signals.append("legal_or_severe_dissatisfaction")
            reasons.append("Contains legal threat, fraud allegation, or severe customer dissatisfaction requiring human de-escalation.")

        # Risk Rule 3: Financial Dispute / Unauthorized Charge / Refund Demands
        if FINANCIAL_RISK_PATTERN.search(customer_message) or (
            classification.intent == "app_store_billing_and_subscriptions" and any(k in text_lower for k in ["refund", "charge", "card", "money", "unauthorized"])
        ):
            risk_signals.append("financial_transaction_risk")
            reasons.append("Involves financial billing, unauthorized transaction, or refund demand requiring secure verification.")

        # Risk Rule 4: Account Security & Credential Recovery
        if SECURITY_RISK_PATTERN.search(customer_message) or (
            classification.intent == "apple_id_and_icloud" and any(k in text_lower for k in ["lock", "locked", "password", "hack", "recover", "2fa"])
        ):
            risk_signals.append("account_security_risk")
            reasons.append("Involves Apple ID credentials, account lockout, or two-factor authentication recovery.")

        # Risk Rule 5: Auto-escalate intents (e.g. general complaints venting)
        if classification.intent in self.auto_escalate_intents:
            risk_signals.append(f"sensitive_intent_{classification.intent}")
            reasons.append(f"Intent '{classification.intent}' is designated for mandatory human review.")

        # Risk Rule 6: Low Classification Confidence or High Ambiguity
        if confidence < self.min_confidence or getattr(classification, "is_uncertain", False):
            risk_signals.append(f"low_intent_confidence_{confidence:.2f}")
            reasons.append(f"Intent confidence ({confidence:.2f}) or confidence margin ({margin:.2f}) is below automation threshold.")

        # Risk Rule 7: Weak or Missing Historical Retrieval Grounding
        if not evidence:
            risk_signals.append("no_retrieval_evidence")
            reasons.append("No historical support precedent found in knowledge base.")
        elif top_similarity < self.min_retrieval_similarity:
            risk_signals.append(f"weak_retrieval_grounding_{top_similarity:.2f}")
            reasons.append(f"Top historical evidence similarity ({top_similarity:.2f}) is below minimum threshold ({self.min_retrieval_similarity:.2f}).")

        # Risk Rule 8: Unclassifiable "other" intent
        if classification.intent == "other":
            risk_signals.append("unclassified_intent_other")
            reasons.append("Query is unclassified or outside standard Apple Support troubleshooting domain.")

        # Final Decision
        if risk_signals:
            return EscalationResult(
                decision=DecisionEnum.ESCALATE_TO_HUMAN,
                reason=" ".join(reasons),
                risk_signals=risk_signals,
                confidence=round(confidence, 4),
                retrieval_strength=round(top_similarity, 4),
                confidence_margin=round(margin, 4)
            )

        # Safe Auto-Handle: Routine issue + High confidence + Strong retrieval evidence + Zero risk signals
        return EscalationResult(
            decision=DecisionEnum.AUTO_HANDLE,
            reason=(
                f"Routine technical support issue ({classification.intent}) classified with {confidence:.2f} confidence "
                f"(margin: {margin:.2f}) and grounded in historical precedent (similarity: {top_similarity:.2f})."
            ),
            risk_signals=[],
            confidence=round(confidence, 4),
            retrieval_strength=round(top_similarity, 4),
            confidence_margin=round(margin, 4)
        )
