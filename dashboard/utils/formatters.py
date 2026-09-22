"""Formatting helpers for metrics, labels, confidence scores, and timestamps."""

from typing import Optional


def format_intent_name(intent_raw: Optional[str]) -> str:
    """Formats snake_case intent identifier into a clean title."""
    if not intent_raw:
        return "Unknown"
    return intent_raw.replace("_", " ").title()


def format_percentage(val: Optional[float], decimals: int = 1) -> str:
    """Formats float [0.0, 1.0] into a clean percentage string."""
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def format_latency(ms: Optional[float]) -> str:
    """Formats latency in milliseconds."""
    if ms is None:
        return "0 ms"
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms / 1000:.2f} s"


def format_score_stars(score: float, max_score: float = 5.0) -> str:
    """Renders star rating text."""
    full_stars = int(score)
    half_star = (score - full_stars) >= 0.5
    stars = "★" * full_stars + ("½" if half_star else "")
    return f"{stars} ({score:.2f}/{max_score:.1f})"
