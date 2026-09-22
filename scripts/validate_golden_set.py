"""Phase 2: Validation script for the Human Golden Evaluation Set.

Verifies:
1. Exactly 200 examples
2. No duplicate IDs
3. No duplicate customer messages
4. No missing / blank intent labels
5. No missing / blank decisions
6. No missing / blank reasons
7. Valid intent values (one of the 9 taxonomy IDs)
8. Valid decision values (AUTO / AUTO_HANDLE or ESCALATE / ESCALATE_TO_HUMAN)

Exits with non-zero status code if validation fails.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.intent_taxonomy import get_all_intent_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VALID_INTENTS = set(get_all_intent_ids())
VALID_DECISIONS = {"AUTO", "AUTO_HANDLE", "ESCALATE", "ESCALATE_TO_HUMAN"}


def validate_golden_set(csv_path: Path, expected_count: int = 200, status_only: bool = False) -> bool:
    """Validates the golden set CSV against all data integrity rules."""
    if not csv_path.exists():
        logger.error(f"Validation failed: Golden set file does not exist at {csv_path}")
        return False

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    total_rows = len(df)
    errors = []

    # 1. Row count check
    if total_rows != expected_count:
        errors.append(f"Expected exactly {expected_count} examples, but found {total_rows}.")

    # 2. Duplicate ID check
    duplicate_ids = df[df["id"].duplicated()]["id"].tolist()
    if duplicate_ids:
        errors.append(f"Found {len(duplicate_ids)} duplicate IDs: {duplicate_ids[:5]}...")

    # 3. Duplicate customer message check
    clean_messages = df["customer_message"].str.strip().str.lower()
    duplicate_msgs = df[clean_messages.duplicated()]["id"].tolist()
    if duplicate_msgs:
        errors.append(f"Found {len(duplicate_msgs)} duplicate customer messages (IDs: {duplicate_msgs[:5]}).")

    # 4. Missing label checks
    unlabeled_intents = df[df["gold_intent"].str.strip() == ""]["id"].tolist()
    unlabeled_decisions = df[df["gold_decision"].str.strip() == ""]["id"].tolist()
    unlabeled_reasons = df[df["gold_reason"].str.strip() == ""]["id"].tolist()

    labeled_count = sum(
        1 for _, r in df.iterrows()
        if r["gold_intent"].strip() and r["gold_decision"].strip() and r["gold_reason"].strip()
    )

    if status_only:
        print("\n" + "=" * 65)
        print("GOLDEN EVALUATION SET STATUS REPORT")
        print("=" * 65)
        print(f"File Path:                {csv_path}")
        print(f"Total Rows:               {total_rows} / {expected_count}")
        print(f"Completed Annotations:    {labeled_count} / {total_rows} ({labeled_count / total_rows:.1%})")
        print(f"Pending Annotations:      {total_rows - labeled_count} / {total_rows}")
        if labeled_count > 0:
            print("\nAnnotated Intent Breakdown:")
            intents = df[df["gold_intent"].str.strip() != ""]["gold_intent"].value_counts()
            for k, v in intents.items():
                print(f"  - {k:<36}: {v} ({v/labeled_count:.1%})")
            print("\nAnnotated Decision Breakdown:")
            decisions = df[df["gold_decision"].str.strip() != ""]["gold_decision"].value_counts()
            for k, v in decisions.items():
                print(f"  - {k:<20}: {v} ({v/labeled_count:.1%})")
        print("=" * 65)
        return labeled_count == expected_count and len(errors) == 0

    # Strict validation errors
    if unlabeled_intents:
        errors.append(f"{len(unlabeled_intents)} rows are missing 'gold_intent' (first unlabelled: {unlabeled_intents[0]}).")
    if unlabeled_decisions:
        errors.append(f"{len(unlabeled_decisions)} rows are missing 'gold_decision' (first unlabelled: {unlabeled_decisions[0]}).")
    if unlabeled_reasons:
        errors.append(f"{len(unlabeled_reasons)} rows are missing 'gold_reason' (first unlabelled: {unlabeled_reasons[0]}).")

    # 5. Invalid intent values
    for idx, row in df.iterrows():
        intent_val = row["gold_intent"].strip()
        if intent_val and intent_val not in VALID_INTENTS:
            errors.append(f"Row {row['id']} has invalid intent '{intent_val}'. Valid: {sorted(list(VALID_INTENTS))}")

    # 6. Invalid decision values
    for idx, row in df.iterrows():
        dec_val = row["gold_decision"].strip()
        if dec_val and dec_val not in VALID_DECISIONS:
            errors.append(f"Row {row['id']} has invalid decision '{dec_val}'. Valid: {sorted(list(VALID_DECISIONS))}")

    # Report results
    if errors:
        logger.error(f"Golden set validation FAILED with {len(errors)} error(s):")
        for err in errors[:10]:
            logger.error(f"  - {err}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more errors.")
        return False

    logger.info(f"Golden set validation PASSED: all {expected_count} examples are fully and validly annotated!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate human golden set evaluation CSV.")
    parser.add_argument("--count", type=int, default=200, help="Expected number of examples (default: 200)")
    parser.add_argument("--status", action="store_true", help="Print labeling progress status without strict error exit")
    args = parser.parse_args()

    config = load_config()
    csv_path = config.data.resolve_path(config.data.golden_set_path)

    is_valid = validate_golden_set(csv_path, expected_count=args.count, status_only=args.status)

    if not is_valid and not args.status:
        print("\n[VALIDATION FAILED] Run 'python scripts/label_golden_set.py' to complete annotations.")
        sys.exit(1)
    elif is_valid:
        print("\n[VALIDATION PASSED] Golden set is complete and verified.")
        sys.exit(0)


if __name__ == "__main__":
    main()
