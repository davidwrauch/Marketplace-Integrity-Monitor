from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import DECISION_THRESHOLD


def apply_decision_layer(
    behavioral_drift: DataFrame,
    model_drift: DataFrame,
    decision_threshold: float = DECISION_THRESHOLD,
) -> DataFrame:
    joined = behavioral_drift.join(
        model_drift.select(
            "review_month",
            "feature_drift_score",
            "prediction_drift_score",
            "model_drift_score",
            "retraining_alert",
        ),
        "review_month",
        "left",
    )

    return (
        joined.fillna(
            {
                "feature_drift_score": 0.0,
                "prediction_drift_score": 0.0,
                "model_drift_score": 0.0,
                "suspicious_review_density": 0.0,
                "behavioral_drift_score": 0.0,
                "avg_anomaly_score": 0.0,
                "max_anomaly_score": 0.0,
            }
        )
        .withColumn(
            "decision_score",
            F.least(
                F.lit(1.0),
                0.35 * F.col("max_anomaly_score")
                + 0.25 * F.col("suspicious_review_density")
                + 0.25 * F.col("behavioral_drift_score")
                + 0.15 * F.col("model_drift_score"),
            ),
        )
        .withColumn(
            "route_to_moderation",
            (F.col("decision_score") >= F.lit(decision_threshold))
            | (
                (F.col("suspicious_review_density") >= F.lit(0.25))
                & (F.col("review_count") >= F.lit(3))
            )
            | (
                (F.col("max_anomaly_score") >= F.lit(0.90))
                & (F.col("behavioral_drift_score") >= F.lit(0.35))
            ),
        )
        .orderBy(F.desc("decision_score"), F.desc("review_count"))
    )


def select_moderation_queue(decisions: DataFrame) -> DataFrame:
    return decisions.where(F.col("route_to_moderation")).select(
        "business_id",
        "business_name",
        "product_title",
        "city",
        "state",
        "review_month",
        "review_count",
        "avg_rating",
        "extreme_rating_share",
        "avg_review_length",
        "avg_anomaly_score",
        "max_anomaly_score",
        "suspicious_review_density",
        "review_volume_change",
        "rating_shift",
        "extreme_rating_share_shift",
        "review_length_shift",
        "behavioral_drift_score",
        "feature_drift_score",
        "prediction_drift_score",
        "model_drift_score",
        "retraining_alert",
        "decision_score",
    )
