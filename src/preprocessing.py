"""Data preprocessing and normalization module for Customer Support on Twitter."""

import re
import html
import unicodedata
from typing import Optional


# Regex patterns for normalization
HANDLE_PATTERN = re.compile(r"@[A-Za-z0-9_]+", re.UNICODE)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.UNICODE)
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")
NON_ASCII_CLEANUP = re.compile(r"[^\x00-\x7F]+", re.UNICODE)


def clean_text(text: Optional[str], replace_handles: bool = True, replace_urls: bool = True) -> str:
    """Cleans and normalizes customer support text.
    
    Steps:
    1. Handle None or empty values gracefully.
    2. HTML unescape (e.g. &amp; -> &, &lt; -> <).
    3. Normalize unicode (NFKD / NFC).
    4. Optionally replace or strip @mentions / handles.
    5. Optionally replace or strip URLs.
    6. Normalize whitespace and trim.
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove or normalize URLs
    if replace_urls:
        text = URL_PATTERN.sub(" ", text)

    # Remove or normalize twitter handles
    if replace_handles:
        text = HANDLE_PATTERN.sub(" ", text)

    # Clean multiple spaces and whitespace
    text = MULTIPLE_SPACES_PATTERN.sub(" ", text).strip()

    return text


def is_usable_message(text: Optional[str], min_length: int = 5) -> bool:
    """Checks whether a message has sufficient semantic content to be processed."""
    if not text:
        return False
    cleaned = clean_text(text)
    # Check if cleaned text has at least min_length alphanumeric characters
    alpha_chars = [c for c in cleaned if c.isalnum()]
    return len(alpha_chars) >= min_length


def extract_inbound_outbound_pair(thread_df) -> Optional[dict]:
    """Helper to structure a clean customer-agent conversation pair from a thread slice."""
    # Used in data loader to construct structured training/retrieval records
    pass
