from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pyspark.sql import Window
from pyspark.sql import functions as F

from src.behavioral_drift import compute_behavioral_drift, compute_business_month_signals
from src.config import AMAZON_CATEGORY, CONTAMINATION, DATASET, DECISION_THRESHOLD
from src.decisions import apply_decision_layer, select_moderation_queue
from src.features import build_review_features
from src.ingest_amazon import build_amazon_products, load_amazon_metadata, load_amazon_reviews
from src.model_drift import compute_model_drift
from src.score_model import score_reviews
from src.spark_session import get_spark
from src.train_model import train_isolation_forest


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local marketplace integrity ML pipeline.")
    parser.add_argument("--raw-data-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--training-months", type=int, default=12)
    parser.add_argument("--contamination", type=float, default=CONTAMINATION)
    parser.add_argument("--decision-threshold", type=float, default=DECISION_THRESHOLD)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--top-reviews-per-business-month", type=int, default=10)
    return parser.parse_args()


def write_parquet(df, path: str | Path) -> None:
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    pdf = df.limit(10000).toPandas()
    pdf.to_csv(output_path / "data.csv", index=False)


def log_count(label: str, df) -> int:
    count = df.count()
    LOGGER.info(f"{label}: {count:,}")
    return count


def main() -> None:
    args = parse_args()
    LOGGER.info("Running dataset: Amazon All_Beauty only")
    spark = get_spark()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    moderation_dir = output_dir / "moderation_queue"
    drift_dir = output_dir / "drift_metrics"
    monitoring_dir = output_dir / "model_monitoring"

    if DATASET != "amazon":
        raise ValueError("This local pipeline is locked to DATASET = amazon.")

    reviews = load_amazon_reviews(spark, args.raw_data_dir, AMAZON_CATEGORY)
    metadata = load_amazon_metadata(spark, args.raw_data_dir, AMAZON_CATEGORY)
    products = build_amazon_products(reviews, AMAZON_CATEGORY, metadata)
    businesses = products.drop("product_id")
    entity_label = "products"

    log_count("raw reviews", reviews)
    log_count(entity_label, businesses)
    LOGGER.info(f"products with non-null product_title: {businesses.where(F.col('product_title').isNotNull()).count():,}")
    LOGGER.info(f"product title sample rows: {businesses.select(F.col('business_id').alias('asin'), 'product_title').limit(5).collect()}")

    features = build_review_features(reviews, businesses).withColumn("dataset", F.lit(DATASET))
    if "product_id" not in features.columns:
        features = features.withColumn("product_id", F.col("business_id"))
    if "review_title" not in features.columns:
        features = features.withColumn("review_title", F.lit(""))
    if "verified_purchase" not in features.columns:
        features = features.withColumn("verified_purchase", F.lit(None).cast("boolean"))
    features = features.persist()
    log_count("review features", features)
    write_parquet(features, processed_dir / "review_features")

    train_isolation_forest(
        features,
        args.model_dir,
        training_months=args.training_months,
        contamination=args.contamination,
        max_train_rows=args.max_train_rows,
    )

    scored_reviews = score_reviews(features, args.model_dir).persist()
    write_parquet(scored_reviews, processed_dir / "scored_reviews")

    business_month = compute_business_month_signals(scored_reviews)
    log_count("business-month rows", business_month)
    behavioral_drift = compute_behavioral_drift(business_month)
    log_count("behavioral drift rows", behavioral_drift)
    write_parquet(behavioral_drift, drift_dir / "behavioral_drift")

    model_drift = compute_model_drift(scored_reviews, args.model_dir)
    log_count("model drift rows", model_drift)
    write_parquet(model_drift, monitoring_dir / "model_drift")

    decisions = apply_decision_layer(behavioral_drift, model_drift, args.decision_threshold)
    queue = select_moderation_queue(decisions).persist()
    queue_count = log_count("moderation queue rows", queue)
    if "product_title" not in queue.columns:
        raise RuntimeError("product_title is missing from final moderation queue output.")
    LOGGER.info(f"moderation queue rows with non-null product_title: {queue.where(F.col('product_title').isNotNull()).count():,}")
    if queue_count == 0:
        LOGGER.warning("Moderation queue is empty. Consider lowering --decision-threshold for exploration.")
    write_parquet(queue, moderation_dir / "business_month_queue")

    queue_signals = queue.select(
        "business_id",
        "review_month",
        "review_count",
        "extreme_rating_share",
        "suspicious_review_density",
        "review_volume_change",
    )
    duplicate_window = Window.partitionBy("business_id", "review_month", "normalized_review_text")
    review_window = Window.partitionBy("business_id", "review_month").orderBy(
        F.desc("fraud_like_score"), F.desc("fraud_signal_count"), F.desc("anomaly_score")
    )
    top_reviews = (
        scored_reviews.join(queue_signals, ["business_id", "review_month"], "inner")
        .withColumn("normalized_review_text", F.lower(F.trim(F.regexp_replace(F.col("text"), r"\s+", " "))))
        .withColumn("near_duplicate_review_count", F.count("*").over(duplicate_window))
        .withColumn(
            "very_short_five_star_signal",
            F.when((F.col("stars") == 5.0) & (F.col("word_count") <= 12), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "high_rating_short_text_signal",
            F.when((F.col("stars") >= 4.0) & (F.col("word_count") <= 20), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "extreme_deviation_signal",
            F.least(F.col("rating_deviation_from_business_average") / F.lit(3.0), F.lit(1.0)),
        )
        .withColumn(
            "duplicate_text_signal",
            F.when((F.col("near_duplicate_review_count") > 1) & (F.col("word_count") >= 4), F.lit(1.0)).otherwise(F.lit(0.0)),
        )
        .withColumn("burst_signal", F.least(F.abs(F.col("review_volume_change")) / F.lit(2.0), F.lit(1.0)))
        .withColumn("extreme_cluster_signal", F.least(F.col("extreme_rating_share") / F.lit(0.75), F.lit(1.0)))
        .withColumn("density_signal", F.least(F.col("suspicious_review_density") / F.lit(0.50), F.lit(1.0)))
        .withColumn("unverified_high_rating_signal", F.when((F.col("stars") >= 4.0) & (F.col("verified_purchase") == F.lit(False)), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("reviewer_extreme_signal", F.least(F.col("reviewer_extreme_rating_share") / F.lit(0.90), F.lit(1.0)))
        .withColumn("bot_text_signal", F.least(F.col("repeated_character_flag") + F.col("text_formatting_anomaly_flag") + F.col("generic_praise_flag") + F.col("no_space_ratio"), F.lit(1.0)))
        .withColumn("synthetic_language_signal", F.when(F.col("synthetic_templated_language_score") >= F.lit(0.35), F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn(
            "low_star_manipulation_signal",
            F.when(
                (F.col("duplicate_text_signal") == 1.0)
                | (F.col("burst_signal") >= 0.50)
                | (F.col("reviewer_extreme_signal") >= 0.75)
                | (F.col("bot_text_signal") >= 0.50)
                | (F.col("synthetic_language_signal") == 1.0),
                F.lit(1.0),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "complaint_only_discount",
            F.when(
                (F.col("stars") <= 2.0)
                & (F.col("duplicate_text_signal") == 0.0)
                & (F.col("burst_signal") < 0.50)
                & (F.col("reviewer_extreme_signal") < 0.75)
                & (F.col("bot_text_signal") < 0.50)
                & (F.col("synthetic_language_signal") == 0.0),
                F.lit(0.25),
            ).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "long_detailed_review_discount",
            F.when((F.col("review_length") >= 900) & (F.col("word_count") >= 140), F.lit(0.15)).otherwise(F.lit(0.0)),
        )
        .withColumn(
            "fraud_signal_count",
            F.col("very_short_five_star_signal")
            + F.col("high_rating_short_text_signal")
            + F.when(F.col("extreme_deviation_signal") >= 0.50, F.lit(1.0)).otherwise(F.lit(0.0))
            + F.col("duplicate_text_signal")
            + F.when(F.col("burst_signal") >= 0.50, F.lit(1.0)).otherwise(F.lit(0.0))
            + F.when(F.col("extreme_cluster_signal") >= 0.75, F.lit(1.0)).otherwise(F.lit(0.0))
            + F.when(F.col("density_signal") >= 0.75, F.lit(1.0)).otherwise(F.lit(0.0))
            + F.col("unverified_high_rating_signal")
            + F.when(F.col("reviewer_extreme_signal") >= 0.75, F.lit(1.0)).otherwise(F.lit(0.0))
            + F.when(F.col("bot_text_signal") >= 0.50, F.lit(1.0)).otherwise(F.lit(0.0))
            + F.col("synthetic_language_signal"),
        )
        .withColumn(
            "fraud_like_score",
            F.greatest(
                F.lit(0.0),
                F.least(
                    F.lit(1.0),
                    0.20 * F.col("very_short_five_star_signal")
                    + 0.15 * F.col("high_rating_short_text_signal")
                    + 0.20 * F.col("extreme_deviation_signal")
                    + 0.15 * F.col("duplicate_text_signal")
                    + 0.10 * F.col("burst_signal")
                    + 0.10 * F.col("extreme_cluster_signal")
                    + 0.10 * F.col("density_signal")
                    + 0.10 * F.col("unverified_high_rating_signal")
                    + 0.10 * F.col("reviewer_extreme_signal")
                    + 0.15 * F.col("bot_text_signal")
                    + 0.05 * F.col("synthetic_templated_language_score")
                    + 0.10 * F.col("anomaly_score")
                    - F.col("long_detailed_review_discount")
                    - F.col("complaint_only_discount"),
                ),
            ),
        )
        .where(F.col("fraud_signal_count") >= F.lit(2.0))
        .where(F.col("fraud_like_score") >= F.lit(0.60))
        .where((F.col("stars") > 2.0) | (F.col("low_star_manipulation_signal") == 1.0))
        .withColumn("review_rank", F.row_number().over(review_window))
        .where(F.col("review_rank") <= F.lit(args.top_reviews_per_business_month))
        .select(
            "business_id",
            "business_name",
            "product_title",
            "product_id",
            "review_month",
            "review_id",
            "user_id",
            "review_date",
            "review_title",
            "verified_purchase",
            "stars",
            "anomaly_score",
            "fraud_like_score",
            "review_length",
            "word_count",
            "useful",
            "funny",
            "cool",
            "is_extreme_rating",
            "rating_deviation_from_business_average",
            "rating_deviation_from_product_average",
            "reviewer_average_rating",
            "reviewer_review_count",
            "reviewer_extreme_rating_share",
            "reviewer_five_star_share",
            "reviewer_one_star_share",
            "uppercase_share",
            "repeated_character_flag",
            "avg_word_length",
            "no_space_ratio",
            "generic_praise_flag",
            "html_break_count",
            "text_formatting_anomaly_flag",
            "sentence_count",
            "avg_sentence_length",
            "sentence_length_variability",
            "first_person_flag",
            "concrete_detail_flag",
            "balanced_pros_cons_flag",
            "specific_experience_flag",
            "repeated_promotional_language_flag",
            "synthetic_templated_language_score",
            "synthetic_language_signal",
            "low_star_manipulation_signal",
            "near_duplicate_review_count",
            "very_short_five_star_signal",
            "high_rating_short_text_signal",
            "extreme_deviation_signal",
            "duplicate_text_signal",
            "burst_signal",
            "extreme_cluster_signal",
            "density_signal",
            "unverified_high_rating_signal",
            "reviewer_extreme_signal",
            "bot_text_signal",
            "complaint_only_discount",
            "fraud_signal_count",
            F.col("clean_review_text").alias("review_text"),
            F.col("clean_review_text").alias("review_text_preview"),
        )
    )
    log_count("top suspicious review rows", top_reviews)
    write_parquet(top_reviews, moderation_dir / "top_suspicious_reviews")

    print("Pipeline complete.")
    print(f"Moderation queue: {moderation_dir / 'business_month_queue'}")
    print(f"Model monitoring: {monitoring_dir / 'model_drift'}")
    print(f"Behavioral drift: {drift_dir / 'behavioral_drift'}")

    spark.stop()


if __name__ == "__main__":
    main()






