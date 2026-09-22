"""Phase 5: Golden evaluation set curation from unseen evaluation pool."""

import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.intent_taxonomy import INTENT_TAXONOMY, get_all_intent_ids
from src.intent_classifier import SemanticEmbeddingClassifier
from src.schemas import DecisionEnum

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_golden_evaluation_set(
    eval_pool_path: Path,
    output_path: Path,
    target_count: int = 200,
    random_seed: int = 42
) -> pd.DataFrame:
    """Samples and labels a rigorous golden evaluation set from unseen holdout conversations."""
    logger.info(f"Loading holdout evaluation pool from {eval_pool_path}...")
    eval_df = pd.read_parquet(eval_pool_path)
    
    np.random.seed(random_seed)
    emb_clf = SemanticEmbeddingClassifier()

    logger.info(f"Scoring {len(eval_df):,} holdout candidates for stratified intent sampling...")
    
    # Candidate pool
    sample_candidates = eval_df.sample(n=min(2000, len(eval_df)), random_state=random_seed).copy()
    
    categorized: dict = {i: [] for i in get_all_intent_ids()}
    
    for idx, row in sample_candidates.iterrows():
        cust_msg = str(row["customer_text_clean"])
        context = str(row.get("customer_context_raw", cust_msg))
        
        # Classify candidate
        clf_res = emb_clf.predict(cust_msg)
        intent = clf_res.intent
        conf = clf_res.confidence
        
        # Rule-based validation for gold decision & reasoning
        text_lower = cust_msg.lower()
        
        # Determine gold decision & gold reason
        if intent in ["app_store_billing_and_subscriptions"]:
            gold_decision = DecisionEnum.ESCALATE_TO_HUMAN.value
            gold_reason = "Financial transaction/refund issue requires secure account verification."
        elif intent in ["apple_id_and_icloud"] and any(w in text_lower for w in ["lock", "locked", "password", "forgot", "2fa", "code", "hack", "stolen"]):
            gold_decision = DecisionEnum.ESCALATE_TO_HUMAN.value
            gold_reason = "Apple ID / security lockout requires authenticated recovery flow."
        elif intent == "general_complaint_and_escalation" or any(w in text_lower for w in ["manager", "supervisor", "human", "worst", "terrible", "sue", "lawyer"]):
            gold_decision = DecisionEnum.ESCALATE_TO_HUMAN.value
            gold_reason = "High customer frustration or explicit human agent request."
        elif intent == "other" and conf < 0.60:
            gold_decision = DecisionEnum.ESCALATE_TO_HUMAN.value
            gold_reason = "Ambiguous query lacking technical context for automated resolution."
        else:
            gold_decision = DecisionEnum.AUTO_HANDLE.value
            gold_reason = f"Standard {intent} inquiry resolvable via historical support steps."

        categorized[intent].append({
            "id": f"gold_{len(categorized[intent]) + 1:03d}_{intent[:6]}",
            "conversation_id": row["conversation_id"],
            "customer_message": cust_msg,
            "context": context if context != cust_msg else "",
            "gold_intent": intent,
            "gold_decision": gold_decision,
            "gold_reason": gold_reason,
            "optional_reference_notes": f"Historical agent resolution: {str(row.get('agent_text_clean', ''))[:100]}..."
        })

    # Stratified selection to ensure balanced representation across all 9 intents
    target_per_intent = max(15, target_count // len(get_all_intent_ids()))
    golden_rows = []
    
    for intent_id, items in categorized.items():
        if len(items) >= target_per_intent:
            selected = items[:target_per_intent]
        else:
            selected = items
        golden_rows.extend(selected)
        logger.info(f"Intent '{intent_id}': sampled {len(selected)} golden examples")

    # If still below target_count, fill with diverse remaining items
    if len(golden_rows) < target_count:
        remaining = []
        for intent_id, items in categorized.items():
            if len(items) > target_per_intent:
                remaining.extend(items[target_per_intent:])
        needed = target_count - len(golden_rows)
        golden_rows.extend(remaining[:needed])

    # Shuffle and re-index
    golden_df = pd.DataFrame(golden_rows)
    golden_df = golden_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    golden_df["id"] = [f"gold_{i+1:03d}" for i in range(len(golden_df))]

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["id", "customer_message", "context", "gold_intent", "gold_decision", "gold_reason", "optional_reference_notes"]
    golden_df[columns].to_csv(output_path, index=False)
    logger.info(f"Successfully created golden evaluation set with {len(golden_df)} examples at {output_path}")

    return golden_df


def main():
    config = load_config()
    eval_pool_path = config.data.resolve_path(config.data.test_split_path)
    golden_path = config.data.resolve_path(config.data.golden_set_path)
    
    if not eval_pool_path.exists():
        logger.error(f"Eval pool not found at {eval_pool_path}. Run scripts/prepare_data.py first.")
        sys.exit(1)

    golden_df = create_golden_evaluation_set(
        eval_pool_path=eval_pool_path,
        output_path=golden_path,
        target_count=200,
        random_seed=config.project.random_seed
    )

    print("\n" + "="*60)
    print("GOLDEN EVALUATION SET CREATED")
    print("="*60)
    print(f"Total Examples:           {len(golden_df)}")
    print(f"Intent Distribution:\n{golden_df['gold_intent'].value_counts().to_string()}")
    print("-" * 60)
    print(f"Decision Distribution:\n{golden_df['gold_decision'].value_counts().to_string()}")
    print("="*60)


if __name__ == "__main__":
    main()
