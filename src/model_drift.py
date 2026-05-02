from __future__ import annotations

import json
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.config import RETRAINING_ALERT_THRESHOLD
from src.features import FEATURE_COLUMNS
from src.train_model import METADATA_FILE


def load_model_metadata(model_dir: str | Path) -> dict:
    with (Path(model_dir) / METADATA_FILE).open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_model_drift(scored_reviews: DataFrame, model_dir: str | Path) -> DataFrame:
    metadata = load_model_metadata(model_dir)
    feature_stats = metadata["feature_stats"]
    score_stats = metadata["score_stats"]

    aggregations = [
        F.count("*").alias("scored_review_count"),
        F.avg("anomaly_score").alias("anomaly_score_mean"),
        F.expr("percentile_approx(anomaly_score, 0.95)").alias("anomaly_score_p95"),
    ]
    for column in FEATURE_COLUMNS:
        aggregations.append(F.avg(column).alias(f"{column}_mean"))

    monthly = scored_reviews.groupBy("review_month").agg(*aggregations)

    feature_drift_terms = []
    for column in FEATURE_COLUMNS:
        baseline = feature_stats[column]
        feature_drift_terms.append(
            F.least(
                F.abs(F.col(f"{column}_mean") - F.lit(float(baseline["mean"])))
                / F.lit(max(float(baseline["std"]), 1e-9)),
                F.lit(5.0),
            )
            / F.lit(5.0)
        )

    feature_drift_score = sum(feature_drift_terms) / F.lit(len(feature_drift_terms))
    prediction_drift_score = F.least(
        (
            F.abs(F.col("anomaly_score_mean") - F.lit(float(score_stats["mean"])))
            / F.lit(max(float(score_stats["std"]), 1e-9))
            + F.abs(F.col("anomaly_score_p95") - F.lit(float(score_stats["p95"])))
            / F.lit(max(float(score_stats["std"]), 1e-9))
        )
        / F.lit(6.0),
        F.lit(1.0),
    )

    return (
        monthly.withColumn("feature_drift_score", feature_drift_score)
        .withColumn("prediction_drift_score", prediction_drift_score)
        .withColumn(
            "model_drift_score",
            F.least(
                F.lit(1.0),
                0.55 * F.col("feature_drift_score") + 0.45 * F.col("prediction_drift_score"),
            ),
        )
        .withColumn(
            "retraining_alert",
            F.col("model_drift_score") >= F.lit(RETRAINING_ALERT_THRESHOLD),
        )
    )
