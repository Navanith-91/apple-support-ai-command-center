"""Phase 1: Sampling 200 real, unlabelled customer interactions for human annotation.

This script samples directly from the unseen evaluation pool (data/processed/eval_pool.parquet)
using a fixed random seed. It guarantees:
1. Zero model prediction leakage (gold_intent, gold_decision, gold_reason are left blank).
2. Exactly 200 unique customer messages.
3. Fully reproducible sampling.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.preprocessing import clean_text, is_usable_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_human_golden_set(
    eval_pool_path: Path,
    output_path: Path,
    target_count: int = 200,
    random_seed: int = 42
) -> pd.DataFrame:
    """Samples 200 distinct, high-quality real customer messages for human annotation."""
    logger.info(f"Loading unseen evaluation pool from {eval_pool_path}...")
    eval_df = pd.read_parquet(eval_pool_path)
    
    # Filter for usable messages
    usable_rows = []
    seen_messages = set()

    for _, row in eval_df.iterrows():
        cust_clean = str(row.get("customer_text_clean", "")).strip()
        if not cust_clean or not is_usable_message(cust_clean, min_length=15):
            continue
        
        # Deduplicate on clean customer message text
        if cust_clean.lower() in seen_messages:
            continue
        
        seen_messages.add(cust_clean.lower())
        usable_rows.append(row)

    usable_df = pd.DataFrame(usable_rows).reset_index(drop=True)
    logger.info(f"Filtered {len(usable_df):,} distinct usable customer interactions from pool of {len(eval_df):,}.")

    if len(usable_df) < target_count:
        raise ValueError(f"Not enough usable rows ({len(usable_df)}) to sample {target_count} examples.")

    # Reproducible random sample
    sampled_df = usable_df.sample(n=target_count, random_state=random_seed).reset_index(drop=True)

    records = []
    for idx, row in sampled_df.iterrows():
        cust_msg = str(row.get("customer_text_clean", "")).strip()
        raw_context = str(row.get("customer_context_raw", "")).strip()
        hist_reply = str(row.get("agent_text_clean", "")).strip()

        # Context is distinct previous context if different from customer message
        context_str = raw_context if raw_context and raw_context != cust_msg else ""
        
        records.append({
            "id": f"gold_{idx + 1:03d}",
            "customer_message": cust_msg,
            "context": context_str,
            "gold_intent": "",      # Left blank for genuine human annotation
            "gold_decision": "",    # Left blank for genuine human annotation
            "gold_reason": "",      # Left blank for genuine human annotation
            "optional_reference_notes": f"Historical agent resolution snippet: {hist_reply[:120]}..." if hist_reply else ""
        })

    golden_df = pd.DataFrame(records)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    golden_df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(f"Successfully created unlabelled golden evaluation set with {len(golden_df)} examples at {output_path}")

    return golden_df


def main():
    parser = argparse.ArgumentParser(description="Create unlabelled 200-sample golden set for human annotation.")
    parser.add_argument("--count", type=int, default=200, help="Number of examples to sample (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    args = parser.parse_args()

    config = load_config()
    eval_pool_path = config.data.resolve_path(config.data.test_split_path)
    output_path = config.data.resolve_path(config.data.golden_set_path)

    if not eval_pool_path.exists():
        logger.error(f"Evaluation pool file not found at {eval_pool_path}. Run scripts/prepare_data.py first.")
        sys.exit(1)

    golden_df = create_human_golden_set(
        eval_pool_path=eval_pool_path,
        output_path=output_path,
        target_count=args.count,
        random_seed=args.seed
    )

    print("\n" + "=" * 60)
    print("UNLABELLED HUMAN GOLDEN EVALUATION SET CREATED")
    print("=" * 60)
    print(f"File Path:                {output_path}")
    print(f"Total Examples Sampled:   {len(golden_df)}")
    print(f"Gold Intent Status:       Blank (0/{len(golden_df)} annotated)")
    print(f"Gold Decision Status:     Blank (0/{len(golden_df)} annotated)")
    print(f"Next Step:                Run 'python scripts/label_golden_set.py' to annotate.")
    print("=" * 60)


if __name__ == "__main__":
    main()
