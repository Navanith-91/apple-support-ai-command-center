"""Data and model loading utilities with Streamlit caching for responsive performance."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import streamlit as st

from src.agent import CustomerSupportAgent
from src.config import load_config

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@st.cache_resource(show_spinner="Initializing AI Agent & Vector Retriever...")
def get_agent() -> CustomerSupportAgent:
    """Loads and caches the full production AI Customer Support Agent pipeline."""
    config = load_config(PROJECT_ROOT / "config" / "config.yaml")
    agent = CustomerSupportAgent(config=config)
    return agent


@st.cache_data(show_spinner="Loading benchmark evaluation results...")
def get_evaluation_results() -> Dict[str, Any]:
    """Loads the comprehensive evaluation results JSON containing baselines, metrics, and failures."""
    eval_path = PROJECT_ROOT / "data" / "processed" / "evaluation_results.json"
    if eval_path.exists():
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner="Loading golden evaluation dataset...")
def get_golden_set() -> pd.DataFrame:
    """Loads the golden dataset with ground truth intents, decisions, and reasons."""
    golden_path = PROJECT_ROOT / "evaluation" / "golden_set.csv"
    if not golden_path.exists():
        golden_path = PROJECT_ROOT / "evaluation" / "golden_set_auto_generated.csv"
    if golden_path.exists():
        return pd.read_csv(golden_path)
    return pd.DataFrame()


@st.cache_data(show_spinner="Loading customer support conversation archive...")
def get_brand_conversations(sample_size: int = 500) -> pd.DataFrame:
    """Loads a representative sample of historical customer-agent conversations for the inbox view."""
    parquet_path = PROJECT_ROOT / "data" / "processed" / "brand_conversations.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if len(df) > sample_size:
            return df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        return df
    return pd.DataFrame()


@st.cache_data
def get_brand_stats() -> pd.DataFrame:
    """Loads brand comparison statistics from data preprocessing."""
    stats_path = PROJECT_ROOT / "data" / "processed" / "brand_stats.csv"
    if stats_path.exists():
        return pd.read_csv(stats_path)
    return pd.DataFrame()
