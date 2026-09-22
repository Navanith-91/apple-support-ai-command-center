"""Phase 2: Interactive CLI Labeling Tool for Human Golden Set Annotation.

Features:
- Incremental save & resume
- Forward/backward navigation ('b' to go back, 's' to skip, 'q' to quit)
- Strict validation of 9 intents, AUTO/ESCALATE decision, and mandatory reason
- Displays progress, message, and conversation context
"""

import sys
import os
from pathlib import Path
import pandas as pd
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.intent_taxonomy import INTENT_TAXONOMY, get_all_intent_ids

# 9 canonical intents
INTENT_CHOICES = [
    ("1", "battery_and_charging", "Battery & Charging Issues (drain, heat, charging cables)"),
    ("2", "app_store_billing_and_subscriptions", "App Store, Billing & Subscriptions (charges, refunds, subscriptions)"),
    ("3", "ios_software_update_bugs", "iOS & Software Update Bugs (post-update crashes, freezes, glitches)"),
    ("4", "general_complaint_and_escalation", "General Complaints & Dissatisfaction (frustration, supervisor request)"),
    ("5", "order_shipping_and_trade_in", "Order, Shipping & Trade-In (deliveries, store pickup, trade-in kit)"),
    ("6", "hardware_audio_and_screen", "Hardware, Screen & Audio Defects (cracked screen, speakers, microphone)"),
    ("7", "apple_id_and_icloud", "Apple ID & iCloud Access (passwords, 2FA lockout, account verification)"),
    ("8", "other", "Other / Unrelated (spam, emojis, outside scope, unanswerable)"),
    ("9", "general_how_to_and_features", "How-To & Feature Inquiries (settings, data transfer, Bluetooth pairing)")
]

INTENT_MAP = {num: i_id for num, i_id, _ in INTENT_CHOICES}
for _, i_id, _ in INTENT_CHOICES:
    INTENT_MAP[i_id] = i_id


def display_guidelines_summary():
    """Prints a quick guide of the 9 intents and decision guidelines."""
    print("=" * 78)
    print("  HUMAN ANNOTATION TOOL - APPLE SUPPORT GOLDEN EVALUATION SET")
    print("=" * 78)
    print("Commands:")
    print("  'q' or 'quit' : Save progress and exit")
    print("  'b' or 'back' : Go back to previous example")
    print("  's' or 'skip' : Skip to next example without modifying")
    print("  'h' or 'help' : Show intent definitions")
    print("-" * 78)


def label_golden_set(csv_path: Path):
    if not csv_path.exists():
        print(f"Error: Golden set file not found at {csv_path}")
        print("Run 'python scripts/create_human_golden_set.py' first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    total = len(df)

    display_guidelines_summary()

    # Find first incomplete record
    start_idx = 0
    for idx, row in df.iterrows():
        if not str(row.get("gold_intent", "")).strip() or not str(row.get("gold_decision", "")).strip():
            start_idx = idx
            break

    idx = start_idx
    while 0 <= idx < total:
        row = df.iloc[idx]
        gold_i = str(row.get("gold_intent", "")).strip()
        gold_d = str(row.get("gold_decision", "")).strip()
        gold_r = str(row.get("gold_reason", "")).strip()
        is_labeled = bool(gold_i and gold_d and gold_r)

        completed_count = sum(1 for _, r in df.iterrows() if str(r.get("gold_intent", "")).strip() and str(r.get("gold_decision", "")).strip())

        print("\n" + "=" * 78)
        print(f"  Example [{idx + 1}/{total}] | ID: {row['id']} | Total Completed: {completed_count}/{total}")
        if is_labeled:
            print(f"  [CURRENT SAVED LABEL]: Intent='{gold_i}' | Decision='{gold_d}'")
            print(f"  [SAVED REASON]: {gold_r}")
        print("-" * 78)
        print(f"CUSTOMER MESSAGE:\n  \"{row['customer_message']}\"\n")
        
        if str(row.get("context", "")).strip():
            print(f"CONTEXT:\n  {row['context']}\n")
        if str(row.get("optional_reference_notes", "")).strip():
            print(f"REFERENCE NOTE:\n  {row['optional_reference_notes']}\n")
        print("-" * 78)

        # 1. Intent Selection
        print("Select Intent:")
        for num, i_id, desc in INTENT_CHOICES:
            current_marker = " <--" if gold_i == i_id else ""
            print(f"  [{num}] {i_id:<36} : {desc}{current_marker}")

        intent_val = None
        while intent_val is None:
            prompt_str = f"Intent (1-9, b=back, s=skip, q=quit) [{gold_i or '1-9'}]: "
            user_input = input(prompt_str).strip()

            if user_input.lower() in ("q", "quit"):
                df.to_csv(csv_path, index=False, encoding="utf-8")
                print(f"\n[INFO] Progress saved to {csv_path}. Exiting.")
                return
            elif user_input.lower() in ("b", "back"):
                if idx > 0:
                    idx -= 1
                    break
                else:
                    print("Already at the first example.")
                    continue
            elif user_input.lower() in ("s", "skip"):
                idx += 1
                break
            elif user_input.lower() in ("h", "help"):
                for _, i_id, desc in INTENT_CHOICES:
                    spec = INTENT_TAXONOMY.get(i_id)
                    print(f"\n* {i_id.upper()}: {spec.description if spec else desc}")
                    if spec:
                        print(f"  Keywords: {', '.join(spec.keywords[:6])}")
                continue
            elif user_input == "" and gold_i:
                intent_val = gold_i
            elif user_input in INTENT_MAP:
                intent_val = INTENT_MAP[user_input]
            else:
                print("Invalid choice! Enter a number 1-9, intent ID, 'b' to go back, or 'q' to quit.")

        if user_input.lower() in ("b", "back", "s", "skip"):
            continue

        # 2. Decision Selection
        print("\nSelect Decision:")
        print(f"  [1] AUTO_HANDLE       : Routine issue with clear troubleshooting path{' <--' if gold_d in ('AUTO', 'AUTO_HANDLE') else ''}")
        print(f"  [2] ESCALATE_TO_HUMAN : Risk, security, billing/refund, anger, or ambiguity{' <--' if gold_d in ('ESCALATE', 'ESCALATE_TO_HUMAN') else ''}")

        decision_val = None
        while decision_val is None:
            default_dec = "1" if gold_d in ("AUTO", "AUTO_HANDLE") else ("2" if gold_d in ("ESCALATE", "ESCALATE_TO_HUMAN") else "")
            dec_input = input(f"Decision (1=AUTO, 2=ESCALATE, b=back, q=quit) [{default_dec}]: ").strip()

            if dec_input.lower() in ("q", "quit"):
                df.to_csv(csv_path, index=False, encoding="utf-8")
                print(f"\n[INFO] Progress saved to {csv_path}. Exiting.")
                return
            elif dec_input.lower() in ("b", "back"):
                break
            elif dec_input in ("1", "auto", "AUTO", "AUTO_HANDLE") or (dec_input == "" and default_dec == "1"):
                decision_val = "AUTO_HANDLE"
            elif dec_input in ("2", "escalate", "ESCALATE", "ESCALATE_TO_HUMAN") or (dec_input == "" and default_dec == "2"):
                decision_val = "ESCALATE_TO_HUMAN"
            else:
                print("Invalid choice! Enter 1 for AUTO or 2 for ESCALATE.")

        if dec_input.lower() in ("b", "back"):
            continue

        # 3. Reason
        reason_val = None
        while not reason_val:
            default_reason = gold_r if gold_r else ""
            prompt_reason = f"Reason for {decision_val} (b=back, q=quit)"
            if default_reason:
                prompt_reason += f" [{default_reason}]"
            prompt_reason += ": "

            reason_input = input(prompt_reason).strip()

            if reason_input.lower() in ("q", "quit"):
                df.to_csv(csv_path, index=False, encoding="utf-8")
                print(f"\n[INFO] Progress saved to {csv_path}. Exiting.")
                return
            elif reason_input.lower() in ("b", "back"):
                break
            elif reason_input == "" and default_reason:
                reason_val = default_reason
            elif len(reason_input) >= 5:
                reason_val = reason_input
            else:
                print("Please provide a meaningful reason (at least 5 characters).")

        if reason_input.lower() in ("b", "back"):
            continue

        # Save to DataFrame
        df.at[idx, "gold_intent"] = intent_val
        df.at[idx, "gold_decision"] = decision_val
        df.at[idx, "gold_reason"] = reason_val

        # Auto-save to CSV immediately
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[OK] Saved record {row['id']}: Intent={intent_val} | Decision={decision_val}")
        idx += 1

    print("\n" + "=" * 78)
    print("ALL 200 EXAMPLES ANNOTATED!")
    print(f"Saved to {csv_path}")
    print("Run 'python scripts/validate_golden_set.py' to verify the dataset integrity.")
    print("=" * 78)


def main():
    config = load_config()
    csv_path = config.data.resolve_path(config.data.golden_set_path)
    label_golden_set(csv_path)


if __name__ == "__main__":
    main()
