"""Phase 5: Independent Human Evaluation Collection Tool.

Allows human evaluators to rate AI customer support responses on a 1-5 scale across:
- Overall Quality (1-5)
- Groundedness (1-5)
- Helpfulness (1-5)
- Tone & Brand Alignment (1-5)
- Actionability (1-5)
- Notes (optional)

Evaluators see the customer message, retrieved evidence, and generated AI reply.
Evaluators do NOT see the LLM judge score before rating.
Progress is saved after every example.
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.agent import CustomerSupportAgent


def get_score_input(prompt: str, current_val: str = "", min_val: int = 1, max_val: int = 5):
    """Prompts for a score between min_val and max_val with validation."""
    while True:
        default_str = f" [{current_val}]" if current_val else ""
        inp = input(f"{prompt}{default_str}: ").strip()
        
        if inp.lower() in ("q", "quit", "b", "back", "s", "skip"):
            return inp.lower()
        if inp == "" and current_val:
            return current_val
        try:
            val = int(inp)
            if min_val <= val <= max_val:
                return str(val)
            print(f"Please enter an integer between {min_val} and {max_val}.")
        except ValueError:
            print(f"Invalid input! Enter a number between {min_val} and {max_val}, 'b' for back, or 'q' to quit.")


def collect_ratings(output_csv: Path, target_samples: int = 30):
    config = load_config()
    golden_path = config.data.resolve_path(config.data.golden_set_path)
    eval_results_path = PROJECT_ROOT / "data" / "processed" / "evaluation_results.json"

    # Load or initialize human_ratings.csv
    if output_csv.exists() and output_csv.stat().st_size > 0:
        ratings_df = pd.read_csv(output_csv, dtype=str).fillna("")
    else:
        ratings_df = pd.DataFrame(columns=[
            "example_id", "customer_message", "retrieved_evidence", "generated_response",
            "human_overall", "human_groundedness", "human_helpfulness", "human_tone",
            "human_actionability", "human_notes"
        ])

    # If rows are missing, populate candidates from golden set or evaluation results
    if len(ratings_df) < target_samples:
        if golden_path.exists():
            gdf = pd.read_csv(golden_path, dtype=str).fillna("")
            agent = CustomerSupportAgent()
            needed = target_samples - len(ratings_df)
            start_offset = len(ratings_df)
            print(f"[INFO] Generating candidate evaluation responses for {needed} sample(s)...")

            new_rows = []
            for _, grow in gdf.iloc[start_offset:start_offset + needed].iterrows():
                msg = grow["customer_message"]
                resp = agent.run(msg)
                ev_str = " | ".join([f"({e['score']:.2f}) {e['agent_reply'][:60]}" for e in resp.evidence[:2]])
                new_rows.append({
                    "example_id": grow["id"],
                    "customer_message": msg,
                    "retrieved_evidence": ev_str,
                    "generated_response": resp.reply,
                    "human_overall": "",
                    "human_groundedness": "",
                    "human_helpfulness": "",
                    "human_tone": "",
                    "human_actionability": "",
                    "human_notes": ""
                })
            ratings_df = pd.concat([ratings_df, pd.DataFrame(new_rows)], ignore_index=True)
            ratings_df.to_csv(output_csv, index=False, encoding="utf-8")

    total = len(ratings_df)
    print("=" * 78)
    print("  INDEPENDENT HUMAN EVALUATION RATING TOOL")
    print("=" * 78)
    print("Rate generated customer support responses on a 1-5 scale:")
    print("  1 = Completely unacceptable / Harmful / Severe Hallucination")
    print("  2 = Poor / Incorrect diagnostic / Weak grounding")
    print("  3 = Acceptable / Generic / Partially helpful")
    print("  4 = Good / Correct troubleshooting / Grounded in evidence")
    print("  5 = Excellent / Perfectly aligned with Apple Support standards")
    print("-" * 78)
    print("Commands: 'b' to go back, 's' to skip, 'q' to save & quit.")
    print("-" * 78)

    # Find starting point
    idx = 0
    for i, r in ratings_df.iterrows():
        if not r["human_overall"].strip():
            idx = i
            break

    while 0 <= idx < total:
        row = ratings_df.iloc[idx]
        cur_ov = str(row.get("human_overall", "")).strip()
        cur_gr = str(row.get("human_groundedness", "")).strip()
        cur_hl = str(row.get("human_helpfulness", "")).strip()
        cur_tn = str(row.get("human_tone", "")).strip()
        cur_ac = str(row.get("human_actionability", "")).strip()
        cur_nt = str(row.get("human_notes", "")).strip()

        completed = sum(1 for _, r in ratings_df.iterrows() if str(r.get("human_overall", "")).strip())

        print("\n" + "=" * 78)
        print(f"  Evaluation Item [{idx + 1}/{total}] | ID: {row['example_id']} | Completed: {completed}/{total}")
        if cur_ov:
            print(f"  [SAVED RATINGS]: Overall={cur_ov}, Groundedness={cur_gr}, Helpful={cur_hl}, Tone={cur_tn}, Actionable={cur_ac}")
        print("-" * 78)
        print(f"CUSTOMER MESSAGE:\n  \"{row['customer_message']}\"\n")
        print(f"RETRIEVED HISTORICAL EVIDENCE:\n  {row['retrieved_evidence']}\n")
        print(f"AI-GENERATED RESPONSE:\n  \"{row['generated_response']}\"\n")
        print("-" * 78)

        # 1. Overall
        ov = get_score_input("1. Overall Quality (1-5)", cur_ov)
        if ov in ("q", "quit"):
            ratings_df.to_csv(output_csv, index=False, encoding="utf-8")
            print(f"\n[INFO] Progress saved to {output_csv}. Exiting.")
            return
        elif ov in ("b", "back"):
            if idx > 0:
                idx -= 1
            continue
        elif ov in ("s", "skip"):
            idx += 1
            continue

        # 2. Groundedness
        gr = get_score_input("2. Groundedness in Evidence (1-5)", cur_gr)
        if gr in ("q", "quit"):
            ratings_df.to_csv(output_csv, index=False, encoding="utf-8")
            return
        elif gr in ("b", "back"):
            continue

        # 3. Helpfulness
        hl = get_score_input("3. Helpfulness (1-5)", cur_hl)
        if hl in ("q", "quit"):
            ratings_df.to_csv(output_csv, index=False, encoding="utf-8")
            return
        elif hl in ("b", "back"):
            continue

        # 4. Tone
        tn = get_score_input("4. Brand Tone & Empathy (1-5)", cur_tn)
        if tn in ("q", "quit"):
            ratings_df.to_csv(output_csv, index=False, encoding="utf-8")
            return
        elif tn in ("b", "back"):
            continue

        # 5. Actionability
        ac = get_score_input("5. Actionability (1-5)", cur_ac)
        if ac in ("q", "quit"):
            ratings_df.to_csv(output_csv, index=False, encoding="utf-8")
            return
        elif ac in ("b", "back"):
            continue

        # 6. Notes
        default_notes = cur_nt
        prompt_notes = "6. Optional Notes" + (f" [{default_notes}]" if default_notes else "") + ": "
        notes_inp = input(prompt_notes).strip()
        if notes_inp.lower() in ("q", "quit"):
            ratings_df.to_csv(output_csv, index=False, encoding="utf-8")
            return
        elif notes_inp == "" and default_notes:
            nt = default_notes
        else:
            nt = notes_inp

        # Save record
        ratings_df.at[idx, "human_overall"] = ov
        ratings_df.at[idx, "human_groundedness"] = gr
        ratings_df.at[idx, "human_helpfulness"] = hl
        ratings_df.at[idx, "human_tone"] = tn
        ratings_df.at[idx, "human_actionability"] = ac
        ratings_df.at[idx, "human_notes"] = nt

        ratings_df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"[OK] Saved rating for {row['example_id']}: Overall={ov}/5")
        idx += 1

    print("\n" + "=" * 78)
    print(f"ALL {total} HUMAN RATINGS COMPLETED!")
    print(f"Saved to {output_csv}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Collect independent human evaluation ratings.")
    parser.add_argument("--samples", type=int, default=30, help="Target number of samples (default: 30)")
    args = parser.parse_args()

    csv_path = PROJECT_ROOT / "evaluation" / "human_ratings.csv"
    collect_ratings(csv_path, target_samples=args.samples)


if __name__ == "__main__":
    main()
