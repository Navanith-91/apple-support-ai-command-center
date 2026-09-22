"""Unit tests for text preprocessing and normalization."""

import pytest
from src.preprocessing import clean_text, is_usable_message


def test_clean_text_strips_handles_and_urls():
    raw = "@AppleSupport my iphone 8 is having battery drain https://t.co/xyz123 &amp; overheating!"
    cleaned = clean_text(raw)
    assert "@AppleSupport" not in cleaned
    assert "https://" not in cleaned
    assert "&amp;" not in cleaned
    assert "&" in cleaned
    assert "battery drain" in cleaned


def test_clean_text_handles_edge_cases():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("    ") == ""
    assert clean_text("Hello   World  \n\r\t") == "Hello World"


def test_is_usable_message():
    assert is_usable_message("My phone is stuck on the Apple logo.") is True
    assert is_usable_message("hi") is False
    assert is_usable_message("???") is False
    assert is_usable_message("") is False
    assert is_usable_message(None) is False
