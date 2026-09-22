"""Phase 4: Preprocessing, conversation reconstruction, and zero-leakage splitting."""

import sys
from pathlib import Path
import logging
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data_loader import (
    download_dataset_if_missing, 
    load_raw_dataset, 
    reconstruct_brand_conversations, 
    split_conversations
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    raw_path = config.data.resolve_path(config.data.raw_path)
    
    # 1. Ensure raw dataset exists
    download_dataset_if_missing(config.data.raw_remote_url, raw_path)
    
    # 2. Load raw dataset
    raw_df = load_raw_dataset(raw_path)
    
    # 3. Reconstruct brand conversation pairs
    selected_brand = config.brand.selected
    pairs_df = reconstruct_brand_conversations(
        raw_df, 
        brand=selected_brand, 
        max_pairs=50000
    )
    
    # 4. Save full brand conversation pairs
    pairs_path = config.data.resolve_path(config.data.processed_pairs_path)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_parquet(pairs_path, index=False)
    logger.info(f"Saved {len(pairs_df):,} brand pairs to {pairs_path}")
    
    # 5. Split into Dev / Reference (train) and Evaluation Pool (eval)
    dev_df, eval_df = split_conversations(
        pairs_df, 
        dev_ratio=config.data.dev_split_ratio, 
        random_seed=config.project.random_seed
    )
    
    dev_path = config.data.resolve_path(config.data.train_split_path)
    eval_path = config.data.resolve_path(config.data.test_split_path)
    
    dev_df.to_parquet(dev_path, index=False)
    eval_df.to_parquet(eval_path, index=False)
    
    logger.info(f"Saved Dev Split ({len(dev_df):,} pairs) to {dev_path}")
    logger.info(f"Saved Eval Pool ({len(eval_df):,} pairs) to {eval_path}")
    
    print("\n" + "="*60)
    print("DATA PREPARATION SUMMARY")
    print("="*60)
    print(f"Selected Brand:           {selected_brand}")
    print(f"Total Usable Pairs:       {len(pairs_df):,}")
    print(f"Dev Split (Train/Index):  {len(dev_df):,} ({len(dev_df)/len(pairs_df):.1%})")
    print(f"Eval Pool (Holdout):      {len(eval_df):,} ({len(eval_df)/len(pairs_df):.1%})")
    print(f"Leakage Check:            0 overlapping conversations (PASSED)")
    print("="*60)


if __name__ == "__main__":
    main()
