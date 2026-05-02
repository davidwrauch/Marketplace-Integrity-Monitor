from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def compute_business_month_signals(scored_reviews: DataFrame) -> DataFrame:
    return scored_reviews.groupBy("business_id", "review_month").agg(
        F.first("business_name", ignorenulls=True).alias("business_name"),
        F.first("product_title", ignorenulls=True).alias("product_title"),
        F.first("city", ignorenulls=True).alias("city"),
        F.first("state", ignorenulls=True).alias("state"),
        F.count("*").alias("review_count"),
        F.avg("stars").alias("avg_rating"),
        F.avg("is_extreme_rating").alias("extreme_rating_share"),
        F.avg("review_length").alias("avg_review_length"),
        F.avg("anomaly_score").alias("avg_anomaly_score"),
        F.max("anomaly_score").alias("max_anomaly_score"),
        F.avg("is_suspicious_review").alias("suspicious_review_density"),
    )


def compute_behavioral_drift(business_month: DataFrame) -> DataFrame:
    window = Window.partitionBy("business_id").orderBy("review_month")

    enriched = (
        business_month.withColumn("prev_review_count", F.lag("review_count").over(window))
        .withColumn("prev_avg_rating", F.lag("avg_rating").over(window))
        .withColumn("prev_extreme_rating_share", F.lag("extreme_rating_share").over(window))
        .withColumn("prev_avg_review_length", F.lag("avg_review_length").over(window))
    )

    return (
        enriched.withColumn(
            "review_volume_change",
            F.when(F.col("prev_review_count").isNull(), F.lit(0.0)).otherwise(
                (F.col("review_count") - F.col("prev_review_count"))
                / F.greatest(F.col("prev_review_count"), F.lit(1))
            ),
        )
        .withColumn(
            "rating_shift",
            F.when(F.col("prev_avg_rating").isNull(), F.lit(0.0)).otherwise(
                F.col("avg_rating") - F.col("prev_avg_rating")
            ),
        )
        .withColumn(
            "extreme_rating_share_shift",
            F.when(F.col("prev_extreme_rating_share").isNull(), F.lit(0.0)).otherwise(
                F.col("extreme_rating_share") - F.col("prev_extreme_rating_share")
            ),
        )
        .withColumn(
            "review_length_shift",
            F.when(F.col("prev_avg_review_length").isNull(), F.lit(0.0)).otherwise(
                (F.col("avg_review_length") - F.col("prev_avg_review_length"))
                / F.greatest(F.col("prev_avg_review_length"), F.lit(1.0))
            ),
        )
        .withColumn(
            "behavioral_drift_score",
            F.least(
                F.lit(1.0),
                0.30 * F.least(F.abs(F.col("review_volume_change")) / F.lit(2.0), F.lit(1.0))
                + 0.25 * F.least(F.abs(F.col("rating_shift")) / F.lit(2.0), F.lit(1.0))
                + 0.25 * F.least(F.abs(F.col("extreme_rating_share_shift")) / F.lit(0.50), F.lit(1.0))
                + 0.20 * F.least(F.abs(F.col("review_length_shift")) / F.lit(1.0), F.lit(1.0)),
            ),
        )
    )
