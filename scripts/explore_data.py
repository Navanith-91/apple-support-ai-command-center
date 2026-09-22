"""Phase 2: Data exploration, brand analysis, and data-driven brand selection."""

import sys
from pathlib import Path
import logging

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data_loader import download_dataset_if_missing, load_raw_dataset
from src.brand_selection import analyze_brand_statistics, print_brand_comparison_table, BRAND_SELECTION_JUSTIFICATION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    raw_path = config.data.resolve_path(config.data.raw_path)
    
    # 1. Download if missing
    download_dataset_if_missing(config.data.raw_remote_url, raw_path)
    
    # 2. Load dataset
    df = load_raw_dataset(raw_path)
    
    # 3. Analyze brand statistics
    logger.info("Computing brand-level statistics across full dataset...")
    stats_df = analyze_brand_statistics(df, top_n=15)
    
    # Save statistics
    processed_dir = config.data.resolve_path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    stats_path = processed_dir / "brand_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"Saved brand statistics to {stats_path}")
    
    # 4. Display comparison table
    print_brand_comparison_table(stats_df)
    
    # 5. Print justification
    print(BRAND_SELECTION_JUSTIFICATION)


if __name__ == "__main__":
    main()
