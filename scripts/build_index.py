"""Builds semantic retrieval index and trains TF-IDF classifier on development data."""

import sys
from pathlib import Path
import logging
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.retriever import HistoricalRetriever
from src.intent_classifier import SemanticEmbeddingClassifier
from evaluation.baselines import TfidfIntentClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    dev_path = config.data.resolve_path(config.data.train_split_path)
    
    if not dev_path.exists():
        logger.error(f"Dev split not found at {dev_path}. Run scripts/prepare_data.py first.")
        sys.exit(1)

    dev_df = pd.read_parquet(dev_path)
    logger.info(f"Loaded {len(dev_df):,} development conversation pairs from {dev_path}")

    # 1. Build and Save Semantic Retrieval Index
    logger.info("Initializing and building HistoricalRetriever semantic index...")
    retriever = HistoricalRetriever(
        model_name=config.retrieval.model_name,
        top_k=config.retrieval.top_k,
        similarity_threshold=config.retrieval.similarity_threshold
    )
    # Index up to 10,000 historical pairs for high retrieval quality and fast index loading
    retriever.build_index(dev_df, max_samples=10000)
    
    index_save_path = config.data.resolve_path(config.data.index_path)
    retriever.save(index_save_path)
    logger.info(f"Retrieval index saved to {index_save_path}")

    # 2. Train TF-IDF Baseline Classifier on prototype pseudo-labels / sampled queries
    logger.info("Training TF-IDF Baseline Classifier...")
    emb_clf = SemanticEmbeddingClassifier(model_name=config.retrieval.model_name)
    
    # Generate labels for a training subset of dev pairs using prototype matching
    sample_train_df = dev_df.sample(n=min(3000, len(dev_df)), random_state=config.project.random_seed).copy()
    texts = sample_train_df["customer_text_clean"].tolist()
    
    logger.info(f"Generating training labels for {len(texts):,} development texts...")
    pseudo_labels = [emb_clf.predict(t).intent for t in texts]
    
    tfidf_clf = TfidfIntentClassifier(random_state=config.project.random_seed)
    tfidf_clf.fit(texts, pseudo_labels)
    
    tfidf_save_path = config.data.resolve_path(config.data.tfidf_model_path)
    tfidf_clf.save(tfidf_save_path)
    logger.info(f"TF-IDF model saved to {tfidf_save_path}")

    print("\n" + "="*60)
    print("INDEXING AND BASELINE TRAINING COMPLETE")
    print("="*60)
    print(f"Indexed Historical Pairs:  {len(retriever.metadata):,}")
    print(f"Index File:                {index_save_path}")
    print(f"TF-IDF Model File:         {tfidf_save_path}")
    print("="*60)


if __name__ == "__main__":
    main()
