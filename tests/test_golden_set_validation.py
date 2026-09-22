"""Tests for Golden Set validation, integrity checks, and duplicate detection."""

import pytest
import pandas as pd
from pathlib import Path

from scripts.validate_golden_set import validate_golden_set


def test_validate_golden_set_valid_data(tmp_path: Path):
    csv_file = tmp_path / "valid_golden_set.csv"
    data = []
    for i in range(10):
        data.append({
            "id": f"gold_{i+1:03d}",
            "customer_message": f"Customer problem description number {i+1} with my phone",
            "context": "",
            "gold_intent": "battery_and_charging",
            "gold_decision": "AUTO_HANDLE",
            "gold_reason": "Routine battery issue with standard diagnostics.",
            "optional_reference_notes": ""
        })
    pd.DataFrame(data).to_csv(csv_file, index=False)

    assert validate_golden_set(csv_file, expected_count=10) is True


def test_validate_golden_set_missing_fields(tmp_path: Path):
    csv_file = tmp_path / "incomplete_golden_set.csv"
    data = [{
        "id": "gold_001",
        "customer_message": "My battery is dying fast",
        "context": "",
        "gold_intent": "",  # Missing
        "gold_decision": "AUTO_HANDLE",
        "gold_reason": "Reason",
        "optional_reference_notes": ""
    }]
    pd.DataFrame(data).to_csv(csv_file, index=False)

    assert validate_golden_set(csv_file, expected_count=1) is False


def test_validate_golden_set_duplicate_ids(tmp_path: Path):
    csv_file = tmp_path / "dup_ids.csv"
    data = [
        {
            "id": "gold_001",
            "customer_message": "Message 1",
            "context": "",
            "gold_intent": "battery_and_charging",
            "gold_decision": "AUTO_HANDLE",
            "gold_reason": "Reason 1",
            "optional_reference_notes": ""
        },
        {
            "id": "gold_001",  # Duplicate ID
            "customer_message": "Message 2",
            "context": "",
            "gold_intent": "hardware_audio_and_screen",
            "gold_decision": "AUTO_HANDLE",
            "gold_reason": "Reason 2",
            "optional_reference_notes": ""
        }
    ]
    pd.DataFrame(data).to_csv(csv_file, index=False)

    assert validate_golden_set(csv_file, expected_count=2) is False


def test_validate_golden_set_duplicate_messages(tmp_path: Path):
    csv_file = tmp_path / "dup_msgs.csv"
    data = [
        {
            "id": "gold_001",
            "customer_message": "Exact same customer text message here",
            "context": "",
            "gold_intent": "battery_and_charging",
            "gold_decision": "AUTO_HANDLE",
            "gold_reason": "Reason 1",
            "optional_reference_notes": ""
        },
        {
            "id": "gold_002",
            "customer_message": "Exact same customer text message here",  # Duplicate message
            "context": "",
            "gold_intent": "battery_and_charging",
            "gold_decision": "AUTO_HANDLE",
            "gold_reason": "Reason 2",
            "optional_reference_notes": ""
        }
    ]
    pd.DataFrame(data).to_csv(csv_file, index=False)

    assert validate_golden_set(csv_file, expected_count=2) is False


def test_validate_golden_set_invalid_intent_or_decision(tmp_path: Path):
    csv_file = tmp_path / "invalid_values.csv"
    data = [{
        "id": "gold_001",
        "customer_message": "Random support message here",
        "context": "",
        "gold_intent": "non_existent_intent_id",  # Invalid
        "gold_decision": "MAYBE",                  # Invalid
        "gold_reason": "Reason",
        "optional_reference_notes": ""
    }]
    pd.DataFrame(data).to_csv(csv_file, index=False)

    assert validate_golden_set(csv_file, expected_count=1) is False
