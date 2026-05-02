from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession


AMAZON_ONLY_MESSAGE = (
    "This local Marketplace Integrity Monitor build is locked to Amazon Reviews 2023: "
    "All_Beauty. Use src.ingest_amazon.load_amazon_reviews and "
    "src.ingest_amazon.build_amazon_products instead."
)


def load_reviews(spark: SparkSession, raw_data_dir: Path) -> DataFrame:
    raise RuntimeError(AMAZON_ONLY_MESSAGE)


def load_businesses(spark: SparkSession, raw_data_dir: Path) -> DataFrame:
    raise RuntimeError(AMAZON_ONLY_MESSAGE)