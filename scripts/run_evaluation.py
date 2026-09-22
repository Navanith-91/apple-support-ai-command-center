"""Phase 13, 16 & 24: Automated benchmark evaluation script comparing Baselines and AI Agent."""

import sys
import json
import logging
from pathlib import Path
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.agent import CustomerSupportAgent
from evaluation.evaluate import EvaluationHarness, print_comparison_table
from evaluation.human_agreement import format_agreement_table
from evaluation.metrics import format_metrics_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    golden_path = config.data.resolve_path(config.data.golden_set_path)
    dev_path = config.data.resolve_path(config.data.train_split_path)
    
    if not golden_path.exists():
        logger.info(f"Golden set not found at {golden_path}. Creating golden set now...")
        from scripts.create_golden_set import main as make_golden
        make_golden()

    # Initialize agent
    logger.info("Initializing CustomerSupportAgent for benchmark evaluation...")
    agent = CustomerSupportAgent(config=config)

    # Run evaluation harness
    harness = EvaluationHarness(golden_set_path=golden_path)
    results = harness.run_all(agent=agent, dev_df_path=dev_path)

    # Save results to JSON
    output_dir = config.data.resolve_path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "evaluation_results.json"
    
    # Clean results for serialization
    serializable = {}
    for k, v in results.items():
        serializable[k] = {}
        for sub_k, sub_v in v.items():
            if isinstance(sub_v, dict) and "per_intent" in sub_v:
                copy_dict = dict(sub_v)
                if isinstance(copy_dict["per_intent"], pd.DataFrame):
                    copy_dict["per_intent"] = copy_dict["per_intent"].to_dict(orient="records")
                if "confusion_matrix" in copy_dict:
                    copy_dict["confusion_matrix"] = copy_dict["confusion_matrix"].tolist()
                serializable[k][sub_k] = copy_dict
            elif isinstance(sub_v, dict):
                serializable[k][sub_k] = sub_v
            elif isinstance(sub_v, list):
                serializable[k][sub_k] = sub_v
            else:
                serializable[k][sub_k] = str(sub_v)

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Full evaluation results saved to {results_path}")

    # Print summary tables
    print("\n" + "="*80)
    print("PHASE 16: TWO BASELINE COMPARISON BENCHMARK RESULTS")
    print("="*80)
    print_comparison_table(results)

    print("\n" + "="*80)
    print("PHASE 15: HUMAN VS LLM JUDGE AGREEMENT STUDY")
    print("="*80)
    print(format_agreement_table(results["ai_agent"]["agreement_study"]))

    print("\n" + "="*80)
    print("LATENCY AND RUNTIME PROFILE")
    print("="*80)
    lat = results["ai_agent"]["latency"]
    print(f"Mean Total Latency:   {lat['mean_latency_ms']:.1f} ms")
    print(f"P95 Total Latency:    {lat['p95_latency_ms']:.1f} ms")
    print(f"Min / Max Latency:    {lat['min_latency_ms']:.1f} ms / {lat['max_latency_ms']:.1f} ms")
    print("="*80)


if __name__ == "__main__":
    main()
