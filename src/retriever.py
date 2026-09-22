"""Semantic historical retrieval module using sentence embeddings and cosine similarity."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from src.schemas import RetrievedEvidence
from src.preprocessing import clean_text

logger = logging.getLogger(__name__)


class HistoricalRetriever:
    """Retrieves top-K historically resolved customer support conversations for grounding."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        similarity_threshold: float = 0.45
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.model = None
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []

    def _get_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ({e}). Falling back to TF-IDF retriever.")
                self.model = None
        return self.model

    def build_index(self, dev_df: pd.DataFrame, max_samples: Optional[int] = 10000) -> "HistoricalRetriever":
        """Builds semantic index from the development/reference conversation pairs."""
        logger.info(f"Building historical retrieval index from {len(dev_df):,} development pairs...")
        
        if max_samples and len(dev_df) > max_samples:
            indexed_df = dev_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        else:
            indexed_df = dev_df.reset_index(drop=True)

        self.metadata = []
        queries = []

        for _, row in indexed_df.iterrows():
            cust_text = row.get("customer_text_clean") or clean_text(row.get("customer_text_raw", ""))
            agent_text = row.get("agent_text_clean") or clean_text(row.get("agent_text_raw", ""))
            
            if not cust_text or not agent_text:
                continue

            self.metadata.append({
                "conversation_id": str(row.get("conversation_id", "")),
                "customer_query": cust_text,
                "agent_reply": agent_text,
                "brand": str(row.get("brand", "AppleSupport")),
                "timestamp": str(row.get("created_at_customer", "")),
                "intent": str(row.get("intent", "unassigned"))
            })
            queries.append(cust_text)

        model = self._get_model()
        if model is not None:
            logger.info(f"Encoding {len(queries):,} customer queries with {self.model_name}...")
            raw_embs = model.encode(queries, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
            self.embeddings = np.array(raw_embs, dtype=np.float32)
        else:
            # Fallback to TF-IDF dense representation
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=5000, sublinear_tf=True)
            self.embeddings = vec.fit_transform(queries).toarray()
            self._tfidf_vec = vec

        logger.info(f"Successfully indexed {len(self.metadata):,} historical support conversations.")
        return self

    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None
    ) -> List[RetrievedEvidence]:
        """Retrieves the top-K most similar historical conversations for a given customer query."""
        if self.embeddings is None or not self.metadata:
            logger.warning("Retriever index is empty.")
            return []

        k = top_k or self.top_k
        min_sim = min_similarity if min_similarity is not None else self.similarity_threshold

        clean_q = clean_text(query)
        if not clean_q:
            return []

        model = self._get_model()
        if model is not None:
            q_emb = model.encode([clean_q], normalize_embeddings=True)
            # Cosine similarity for normalized vectors is the dot product
            scores = np.dot(self.embeddings, q_emb.T).flatten()
        else:
            q_emb = self._tfidf_vec.transform([clean_q]).toarray()
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = q_emb / norm
            scores = np.dot(self.embeddings, q_emb.T).flatten()

        # Top K indices sorted descending
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            meta = self.metadata[idx]
            results.append(RetrievedEvidence(
                conversation_id=meta["conversation_id"],
                score=round(score, 4),
                customer_query=meta["customer_query"],
                agent_reply=meta["agent_reply"],
                intent=meta.get("intent"),
                timestamp=meta.get("timestamp")
            ))

        return results

    def is_evidence_sufficient(self, evidence: List[RetrievedEvidence], threshold: Optional[float] = None) -> bool:
        """Evaluates whether retrieved evidence is strong enough to safely ground an automated response."""
        thresh = threshold if threshold is not None else self.similarity_threshold
        if not evidence:
            return False
        return evidence[0].score >= thresh

    def evaluate_retrieval_benchmarks(self, sample_queries: List[str]) -> Dict[str, Any]:
        """Calculates mean top-1 similarity, mean top-K similarity, and sufficiency rate."""
        if not sample_queries or self.embeddings is None:
            return {
                "mean_top1_similarity": 0.0,
                "mean_topk_similarity": 0.0,
                "evidence_sufficiency_rate": 0.0,
                "evaluated_queries": 0
            }

        top1_scores = []
        topk_means = []
        sufficient_count = 0

        for q in sample_queries:
            evs = self.retrieve(q, top_k=self.top_k)
            if evs:
                top1_scores.append(evs[0].score)
                topk_means.append(np.mean([e.score for e in evs]))
                if evs[0].score >= self.similarity_threshold:
                    sufficient_count += 1
            else:
                top1_scores.append(0.0)
                topk_means.append(0.0)

        n = len(sample_queries)
        return {
            "mean_top1_similarity": round(float(np.mean(top1_scores)), 4) if top1_scores else 0.0,
            "mean_topk_similarity": round(float(np.mean(topk_means)), 4) if topk_means else 0.0,
            "evidence_sufficiency_rate": round(float(sufficient_count / n), 4) if n > 0 else 0.0,
            "evaluated_queries": n
        }

    def save(self, filepath: Path) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "embeddings": self.embeddings,
                "metadata": self.metadata,
                "model_name": self.model_name,
                "top_k": self.top_k,
                "similarity_threshold": self.similarity_threshold
            }, f)
        logger.info(f"Saved retrieval index ({len(self.metadata):,} items) to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "HistoricalRetriever":
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Retrieval index file not found at {filepath}")
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        instance = cls(
            model_name=data.get("model_name", "all-MiniLM-L6-v2"),
            top_k=data.get("top_k", 3),
            similarity_threshold=data.get("similarity_threshold", 0.45)
        )
        instance.embeddings = data["embeddings"]
        instance.metadata = data["metadata"]
        logger.info(f"Loaded retrieval index with {len(instance.metadata):,} items from {filepath}")
        return instance
