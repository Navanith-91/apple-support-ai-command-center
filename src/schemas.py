"""Pydantic schemas and data models for the AI Customer Support Agent."""

from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field


class DecisionEnum(str, Enum):
    """Decision for automated handling vs human escalation."""
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


class RetrievedEvidence(BaseModel):
    """Metadata and content of historical support conversation retrieved for grounding."""
    conversation_id: str = Field(..., description="ID or tweet ID of historical conversation")
    score: float = Field(..., description="Cosine similarity score [0.0 - 1.0]")
    customer_query: str = Field(..., description="Customer's historical inbound message")
    agent_reply: str = Field(..., description="Brand support agent's historical resolution reply")
    intent: Optional[str] = Field(None, description="Classified or assigned intent if available")
    timestamp: Optional[str] = Field(None, description="Timestamp of historical interaction")


class IntentClassificationResult(BaseModel):
    """Result from the intent classification stage with full explainability."""
    intent: str = Field(..., description="Predicted intent label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]")
    reason: str = Field(..., description="Explanation of why this intent was selected")
    scores: Optional[Dict[str, float]] = Field(default_factory=dict, description="Probabilities/scores across all intents")
    top_alternatives: Optional[List[Tuple[str, float]]] = Field(default_factory=list, description="Top runner-up alternative intents and scores")
    confidence_margin: Optional[float] = Field(default=0.0, description="Difference between top-1 and top-2 predictions")
    semantic_similarity: Optional[float] = Field(default=0.0, description="Raw cosine similarity score with prototype")
    keyword_evidence: Optional[List[str]] = Field(default_factory=list, description="Domain keywords detected in customer input")
    is_uncertain: Optional[bool] = Field(default=False, description="Whether the classifier detected high ambiguity or low confidence")


class GenerationResult(BaseModel):
    """Result from the response generation stage."""
    reply: str = Field(..., description="Drafted customer support response")
    grounded: bool = Field(default=True, description="Whether response is directly grounded in retrieved historical evidence")
    evidence_used: List[str] = Field(default_factory=list, description="IDs or snippets of evidence used")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in grounded response")
    uncertainty: Optional[str] = Field(None, description="Explanation of uncertainty if evidence is weak")
    safety_notes: List[str] = Field(default_factory=list, description="Safety guardrails and policy constraints applied")
    potential_risk: Optional[str] = Field(None, description="Risk warning or hallucination check result")


class EscalationResult(BaseModel):
    """Result from the escalation policy engine with full risk attribution."""
    decision: DecisionEnum = Field(..., description="AUTO_HANDLE or ESCALATE_TO_HUMAN")
    reason: str = Field(..., description="Detailed explanation of the escalation decision")
    risk_signals: List[str] = Field(default_factory=list, description="List of identified risk flags")
    confidence: float = Field(default=0.0, description="Classifier confidence score")
    retrieval_strength: float = Field(default=0.0, description="Max retrieval cosine similarity score")
    confidence_margin: float = Field(default=0.0, description="Confidence margin between top-1 and top-2 intents")


class AgentResponse(BaseModel):
    """Final unified response output from the Customer Support Agent."""
    intent: str = Field(..., description="Classified intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Intent classification confidence")
    reply: str = Field(..., description="Generated customer support response")
    decision: str = Field(..., description="Decision: AUTO_HANDLE or ESCALATE_TO_HUMAN")
    reason: str = Field(..., description="Reason for decision or escalation explanation")
    evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved historical evidence pairs")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Execution metrics (latency, model)")


class GoldenExample(BaseModel):
    """A manually curated or validated golden evaluation example."""
    id: str
    customer_message: str
    context: Optional[str] = ""
    gold_intent: str
    gold_decision: str
    gold_reason: str
    optional_reference_notes: Optional[str] = ""


class LLMJudgeScore(BaseModel):
    """Evaluation score output from the LLM Judge rubric (1-5 scale)."""
    correctness: int = Field(..., ge=1, le=5, description="Factual and logical correctness (1-5)")
    groundedness: int = Field(..., ge=1, le=5, description="Supported by historical evidence without hallucination (1-5)")
    helpfulness: int = Field(..., ge=1, le=5, description="Actionable and clear steps for customer (1-5)")
    brand_alignment: int = Field(..., ge=1, le=5, description="Tone matches brand support voice (1-5)")
    safety: int = Field(..., ge=1, le=5, description="Safe and does not leak private info or make false promises (1-5)")
    hallucination: int = Field(..., ge=1, le=5, description="Freedom from hallucination (5 = zero hallucination, 1 = severe hallucination)")
    overall: float = Field(..., ge=1.0, le=5.0, description="Overall weighted score (1-5)")
    reason: str = Field(..., description="Detailed justification for scores")
