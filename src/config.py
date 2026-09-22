"""Configuration loader for the Customer Support AI Agent."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field


# Root directory of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class IntentDefinition(BaseModel):
    id: str
    name: str
    description: str


class TaxonomyConfig(BaseModel):
    intents: List[IntentDefinition]


class ProjectConfig(BaseModel):
    name: str = "Customer Support AI Agent"
    version: str = "1.0.0"
    random_seed: int = 42


class BrandConfig(BaseModel):
    selected: str = "AppleSupport"
    description: str = ""


class DataConfig(BaseModel):
    raw_remote_url: str = "https://huggingface.co/datasets/gorkemsevinc/Customer_Support_on_Twitter/resolve/main/data/train-00000-of-00001.parquet"
    raw_path: str = "data/raw/Customer_Support_on_Twitter.parquet"
    processed_pairs_path: str = "data/processed/brand_conversations.parquet"
    train_split_path: str = "data/processed/dev_split.parquet"
    test_split_path: str = "data/processed/eval_pool.parquet"
    golden_set_path: str = "evaluation/golden_set.csv"
    index_path: str = "data/processed/retrieval_index.pkl"
    tfidf_model_path: str = "data/processed/tfidf_classifier.pkl"
    dev_split_ratio: float = 0.85

    def resolve_path(self, relative_path: str) -> Path:
        """Resolves relative path against PROJECT_ROOT."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return PROJECT_ROOT / relative_path


class RetrievalConfig(BaseModel):
    model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 3
    similarity_threshold: float = 0.45
    use_tfidf_fallback: bool = True


class ClassifierConfig(BaseModel):
    type: str = "hybrid"  # tfidf, embedding, hybrid, llm
    confidence_threshold: float = 0.65


class GenerationConfig(BaseModel):
    provider: str = "offline"  # offline, gemini, openai
    model: str = "gemini-1.5-flash"
    temperature: float = 0.2
    max_tokens: int = 250


class EscalationConfig(BaseModel):
    min_confidence: float = 0.65
    min_retrieval_similarity: float = 0.45
    auto_escalate_intents: List[str] = Field(default_factory=list)
    risk_keywords: List[str] = Field(default_factory=list)


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    brand: BrandConfig = Field(default_factory=BrandConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    taxonomy: TaxonomyConfig
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    classifier: ClassifierConfig = Field(default_factory=ClassifierConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load config from YAML and overlay with environment variables."""
    if config_path is None:
        cfg_file = PROJECT_ROOT / "config" / "config.yaml"
    else:
        cfg_file = Path(config_path)
        if not cfg_file.is_absolute():
            cfg_file = PROJECT_ROOT / cfg_file

    if not cfg_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    # Environment variable overrides
    if "SELECTED_BRAND" in os.environ:
        raw_data["brand"]["selected"] = os.environ["SELECTED_BRAND"]
    if "RAW_DATA_PATH" in os.environ:
        raw_data["data"]["raw_path"] = os.environ["RAW_DATA_PATH"]
    if "GOLDEN_SET_PATH" in os.environ:
        raw_data["data"]["golden_set_path"] = os.environ["GOLDEN_SET_PATH"]
    if "INDEX_PATH" in os.environ:
        raw_data["data"]["index_path"] = os.environ["INDEX_PATH"]
    if "LLM_PROVIDER" in os.environ:
        raw_data["generation"]["provider"] = os.environ["LLM_PROVIDER"]
    if "LLM_MODEL" in os.environ:
        raw_data["generation"]["model"] = os.environ["LLM_MODEL"]

    return AppConfig(**raw_data)
