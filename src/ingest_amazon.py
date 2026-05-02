from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


AMAZON_CATEGORY = "All_Beauty"
AMAZON_REVIEW_FILE = Path("amazon") / "All_Beauty.jsonl.gz"
AMAZON_METADATA_FILE = Path("amazon") / "meta_All_Beauty.jsonl.gz"
LOGGER = logging.getLogger(__name__)


def _col_or_null(df: DataFrame, name: str, dtype: str = "string"):
    if name in df.columns:
        return F.col(name)
    return F.lit(None).cast(dtype)


def _review_path(raw_data_dir: str | Path) -> Path:
    path = Path(raw_data_dir) / AMAZON_REVIEW_FILE
    if not path.exists():
        raise FileNotFoundError(
            "Missing Amazon Reviews 2023 All_Beauty file. Expected exactly: "
            f"{path}"
        )
    return path


def _metadata_path(raw_data_dir: str | Path) -> Path:
    return Path(raw_data_dir) / AMAZON_METADATA_FILE


def _validate_category(category: str) -> None:
    if category != AMAZON_CATEGORY:
        raise ValueError(
            f"This local portfolio pipeline is locked to Amazon {AMAZON_CATEGORY}; got {category}."
        )


def load_amazon_metadata(
    spark: SparkSession,
    raw_data_dir: str | Path,
    category: str = AMAZON_CATEGORY,
) -> DataFrame | None:
    _validate_category(category)
    path = _metadata_path(raw_data_dir)
    if not path.exists():
        LOGGER.warning(f"Amazon metadata file not found at {path}; continuing without metadata.")
        return None

    try:
        raw = spark.read.text(str(path))
        LOGGER.info(f"Amazon metadata raw rows read: {raw.count():,}")

        # Amazon metadata uses parent_asin as the product/listing key in many categories.
        asin = F.coalesce(
            F.get_json_object(F.col("value"), "$.parent_asin"),
            F.get_json_object(F.col("value"), "$.asin"),
        )
        title = F.coalesce(
            F.get_json_object(F.col("value"), "$.title"),
            F.get_json_object(F.col("value"), "$.name"),
            F.get_json_object(F.col("value"), "$.product_title"),
            F.get_json_object(F.col("value"), "$.item_name"),
        )
        metadata = raw.select(
            asin.alias("asin"),
            title.alias("product_title"),
        )
        LOGGER.info(f"Amazon metadata rows with non-null asin: {metadata.where(F.col('asin').isNotNull()).count():,}")
        LOGGER.info(
            f"Amazon metadata rows with non-null product_title: "
            f"{metadata.where(F.col('product_title').isNotNull()).count():,}"
        )
        LOGGER.info(
            "Amazon metadata sample asin/product_title rows: "
            f"{metadata.where(F.col('asin').isNotNull()).select('asin', 'product_title').limit(5).collect()}"
        )
        return metadata.where(F.col("asin").isNotNull()).dropDuplicates(["asin"])
    except Exception as exc:
        LOGGER.warning(f"Amazon metadata could not be loaded from {path}; continuing without metadata. {exc}")
        return None


def load_amazon_reviews(
    spark: SparkSession,
    raw_data_dir: str | Path,
    category: str = AMAZON_CATEGORY,
) -> DataFrame:
    _validate_category(category)
    raw = spark.read.json(str(_review_path(raw_data_dir)))
    product_id = F.coalesce(_col_or_null(raw, "parent_asin"), _col_or_null(raw, "asin"))
    timestamp_seconds = F.when(
        _col_or_null(raw, "timestamp", "double").cast("double") > F.lit(9_999_999_999),
        _col_or_null(raw, "timestamp", "double").cast("double") / F.lit(1000.0),
    ).otherwise(_col_or_null(raw, "timestamp", "double").cast("double"))
    review_datetime = F.to_timestamp(F.from_unixtime(timestamp_seconds.cast("long")))
    review_text = F.coalesce(_col_or_null(raw, "text"), F.lit(""))
    review_title = F.coalesce(_col_or_null(raw, "title"), F.lit(""))

    return (
        raw.select(
            F.sha2(
                F.concat_ws("||", product_id, _col_or_null(raw, "user_id"), _col_or_null(raw, "timestamp"), review_text),
                256,
            ).alias("review_id"),
            _col_or_null(raw, "user_id").alias("user_id"),
            product_id.alias("business_id"),
            product_id.alias("product_id"),
            F.lit(category).alias("category"),
            _col_or_null(raw, "asin").alias("asin"),
            _col_or_null(raw, "parent_asin").alias("parent_asin"),
            _col_or_null(raw, "rating", "double").cast("double").alias("stars"),
            F.coalesce(_col_or_null(raw, "helpful_vote", "double").cast("double"), F.lit(0.0)).alias("useful"),
            F.lit(0.0).alias("funny"),
            F.lit(0.0).alias("cool"),
            review_text.alias("text"),
            review_title.alias("review_title"),
            F.coalesce(_col_or_null(raw, "verified_purchase", "boolean").cast("boolean"), F.lit(False)).alias(
                "verified_purchase"
            ),
            review_datetime.alias("review_timestamp"),
            review_datetime.alias("review_datetime"),
        )
        .where(F.col("business_id").isNotNull())
        .where(F.col("stars").isNotNull())
        .where(F.col("review_timestamp").isNotNull())
    )


def build_amazon_products(
    reviews: DataFrame,
    category: str = AMAZON_CATEGORY,
    metadata: DataFrame | None = None,
) -> DataFrame:
    _validate_category(category)
    products = reviews.groupBy("business_id").agg(
        F.first("product_id", ignorenulls=True).alias("product_id"),
        F.first("product_id", ignorenulls=True).alias("fallback_product_id"),
        F.avg("stars").alias("listed_business_stars"),
        F.count("*").cast("long").alias("listed_review_count"),
    )
    if metadata is not None:
        products = products.join(metadata, products.business_id == metadata.asin, "left").drop("asin")
    else:
        products = products.withColumn("product_title", F.lit(None).cast("string"))

    LOGGER.info(
        f"Amazon products with non-null product_title after metadata join: "
        f"{products.where(F.col('product_title').isNotNull()).count():,}"
    )
    LOGGER.info(
        "Amazon product sample asin/product_title rows after metadata join: "
        f"{products.select(F.col('business_id').alias('asin'), 'product_title').limit(5).collect()}"
    )

    products = products.withColumn("category", F.lit(category)).withColumn("average_rating", F.lit(None).cast("double"))

    return (
        products.withColumn("product_title", F.col("product_title"))
        .withColumn("business_name", F.col("product_title"))
        .withColumn("city", F.lit(""))
        .withColumn("state", F.lit(""))
        .withColumn("categories", F.coalesce(F.col("category"), F.lit(category)))
        .drop("fallback_product_id")
    )