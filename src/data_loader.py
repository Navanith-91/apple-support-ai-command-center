"""Data loading, download, conversation reconstruction, and zero-leakage splitting."""

import re
import urllib.request
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.preprocessing import clean_text, is_usable_message

logger = logging.getLogger(__name__)


def download_dataset_if_missing(url: str, destination_path: Path) -> Path:
    """Downloads the dataset parquet file from remote source if not present locally."""
    destination_path = Path(destination_path)
    if destination_path.exists() and destination_path.stat().st_size > 1024 * 1024:
        logger.info(f"Dataset already exists locally at {destination_path} ({destination_path.stat().st_size / 1e6:.1f} MB)")
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading dataset from {url} to {destination_path}...")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response:
        total_size = int(response.info().get("Content-Length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading dataset") as pbar:
            with open(destination_path, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))

    logger.info(f"Successfully downloaded dataset to {destination_path}")
    return destination_path


def load_raw_dataset(path: Path) -> pd.DataFrame:
    """Loads raw dataset parquet file into a pandas DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path}")
    logger.info(f"Loading dataset from {path}...")
    df = pd.read_parquet(path)
    logger.info(f"Loaded dataset with {len(df):,} conversations and columns: {list(df.columns)}")
    return df


def parse_conversation_turns(conversation_text: str) -> Tuple[str, str, str]:
    """Parses customer turns and support agent turns from the multi-turn conversation string.
    
    Returns:
    - customer_initial_query: First customer message
    - customer_full_context: All customer messages combined
    - agent_resolution: First support response
    """
    lines = conversation_text.strip().split("\n")
    cust_turns = []
    agent_turns = []

    current_role = None
    current_msg = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Customer:"):
            if current_role == "Customer":
                cust_turns.append(" ".join(current_msg))
            elif current_role == "Support":
                agent_turns.append(" ".join(current_msg))
            current_role = "Customer"
            current_msg = [stripped[len("Customer:"):].strip()]
        elif stripped.startswith("Support:"):
            if current_role == "Customer":
                cust_turns.append(" ".join(current_msg))
            elif current_role == "Support":
                agent_turns.append(" ".join(current_msg))
            current_role = "Support"
            current_msg = [stripped[len("Support:"):].strip()]
        else:
            if current_msg is not None:
                current_msg.append(stripped)

    if current_role == "Customer":
        cust_turns.append(" ".join(current_msg))
    elif current_role == "Support":
        agent_turns.append(" ".join(current_msg))

    cust_initial = cust_turns[0] if cust_turns else ""
    cust_full = "\n".join(cust_turns) if cust_turns else ""
    agent_res = agent_turns[0] if agent_turns else ""

    return cust_initial, cust_full, agent_res


def reconstruct_brand_conversations(
    raw_df: pd.DataFrame, 
    brand: str = "AppleSupport",
    max_pairs: Optional[int] = 50000
) -> pd.DataFrame:
    """Filters conversations for the selected brand and extracts clean query -> reply pairs."""
    logger.info(f"Filtering and reconstructing pairs for brand '{brand}'...")
    
    brand_df = raw_df[raw_df["company"] == brand].copy()
    logger.info(f"Found {len(brand_df):,} total conversations for {brand}")

    pairs: List[Dict] = []
    for _, row in brand_df.iterrows():
        conv_id = str(row["conversation_id"])
        raw_conv = str(row["conversation"])

        cust_initial, cust_full, agent_res = parse_conversation_turns(raw_conv)

        if not is_usable_message(cust_initial) or not is_usable_message(agent_res):
            continue

        cust_clean = clean_text(cust_initial)
        agent_clean = clean_text(agent_res)

        if not is_usable_message(cust_clean, min_length=4) or not is_usable_message(agent_clean, min_length=4):
            continue

        pairs.append({
            "conversation_id": conv_id,
            "brand": brand,
            "customer_text_raw": cust_initial,
            "customer_context_raw": cust_full,
            "agent_text_raw": agent_res,
            "customer_text_clean": cust_clean,
            "agent_text_clean": agent_clean,
        })

        if max_pairs and len(pairs) >= max_pairs:
            break

    pairs_df = pd.DataFrame(pairs)
    logger.info(f"Successfully constructed {len(pairs_df):,} clean pairs for {brand}")
    return pairs_df


def split_conversations(
    pairs_df: pd.DataFrame, 
    dev_ratio: float = 0.85, 
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits conversations at conversation_id level to guarantee ZERO leakage between reference/train and eval pool."""
    logger.info("Performing conversation-level train/eval split...")
    
    shuffled_df = pairs_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(shuffled_df) * dev_ratio)

    dev_df = shuffled_df.iloc[:split_idx].copy().reset_index(drop=True)
    eval_df = shuffled_df.iloc[split_idx:].copy().reset_index(drop=True)

    # Double check zero overlap
    overlap = set(dev_df["conversation_id"]).intersection(set(eval_df["conversation_id"]))
    assert len(overlap) == 0, f"Critical error: Data leakage detected! Overlapping conversation IDs: {len(overlap)}"

    logger.info(f"Split complete: Dev/Reference Set={len(dev_df):,} pairs, Eval Pool={len(eval_df):,} pairs")
    return dev_df, eval_df
