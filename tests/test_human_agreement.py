"""Tests for Human vs LLM Judge agreement calculations and pending state handling."""

import pytest
from evaluation.human_agreement import compute_judge_agreement_statistics, format_agreement_table


def test_agreement_statistics_perfect_correlation():
    human_scores = [5.0, 4.0, 3.0, 2.0, 1.0]
    llm_scores = [5.0, 4.0, 3.0, 2.0, 1.0]

    stats = compute_judge_agreement_statistics(human_scores, llm_scores)
    assert stats["status"] == "completed"
    assert stats["sample_size"] == 5
    assert stats["exact_agreement"] == 1.0
    assert stats["adjacent_agreement_plus_minus_1"] == 1.0
    assert stats["mean_absolute_difference"] == 0.0
    assert stats["pearson_correlation"] == 1.0
    assert stats["quadratic_weighted_kappa"] == 1.0


def test_agreement_statistics_offset():
    human_scores = [5.0, 4.0, 3.0, 2.0]
    llm_scores = [4.0, 3.0, 2.0, 1.0]

    stats = compute_judge_agreement_statistics(human_scores, llm_scores)
    assert stats["exact_agreement"] == 0.0
    assert stats["adjacent_agreement_plus_minus_1"] == 1.0
    assert stats["mean_absolute_difference"] == 1.0
    assert stats["pearson_correlation"] == 1.0  # Linear shift has r=1.0


def test_agreement_statistics_empty_pending_state():
    stats = compute_judge_agreement_statistics([], [])
    assert stats["status"] == "pending"
    assert stats["sample_size"] == 0
    assert stats["exact_agreement"] is None

    table_md = format_agreement_table(stats)
    assert "pending" in table_md.lower()
