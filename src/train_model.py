from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CONTAMINATION, MIN_TRAINING_ROWS
from src.features import FEATURE_COLUMNS


MODEL_FILE = "isolation_forest.joblib"
METADATA_FILE = "model_metadata.json"
LOGGER = logging.getLogger(__name__)


def training_cutoff_date(
    features: DataFrame,
    training_months: int,
    min_training_rows: int = MIN_TRAINING_ROWS,
) -> str:
    monthly_counts = (
        features.where(F.col("review_month").isNotNull())
        .groupBy("review_month")
        .count()
        .orderBy("review_month")
    )
    if monthly_counts.limit(1).count() == 0:
        raise ValueError("No review months found in feature data.")

    observed_training_months = max(training_months, 24)
    months = [row["review_month"] for row in monthly_counts.select("review_month").limit(observed_training_months).collect()]
    if not months:
        raise ValueError("No review months found in feature data.")

    cutoff = features.select(F.add_months(F.lit(months[-1]), 1).alias("cutoff")).first()["cutoff"]
    rows_before_cutoff = features.where(F.col("review_month") < F.lit(cutoff.isoformat())).count()
    if rows_before_cutoff >= min_training_rows:
        return cutoff.isoformat()

    cumulative_rows = 0
    fallback_month = None
    total_rows = 0
    for row in monthly_counts.collect():
        total_rows += row["count"]
        cumulative_rows += row["count"]
        if cumulative_rows >= min_training_rows:
            fallback_month = row["review_month"]
            break

    if fallback_month is None:
        raise ValueError(
            f"Training data is too small: {total_rows:,} total rows found. "
            f"Need at least {min_training_rows:,}; increase the sample fraction or check raw data."
        )

    fallback_cutoff = features.select(
        F.add_months(F.lit(fallback_month), 1).alias("cutoff")
    ).first()["cutoff"]
    LOGGER.warning(
        "First %s observed months had only %s training rows; using cutoff %s to reach at least %s rows.",
        observed_training_months,
        f"{rows_before_cutoff:,}",
        fallback_cutoff.isoformat(),
        f"{min_training_rows:,}",
    )
    return fallback_cutoff.isoformat()


def _collect_training_frame(
    features: DataFrame,
    cutoff_date: str,
    max_train_rows: int | None,
    random_seed: int,
    min_training_rows: int,
) -> pd.DataFrame:
    train_df = features.where(F.col("review_month") < F.lit(cutoff_date)).select(*FEATURE_COLUMNS)
    total = train_df.count()
    LOGGER.info(f"training rows before optional sampling: {total:,}")
    if total < min_training_rows:
        raise ValueError(
            f"Training data is too small: {total:,} rows found before {cutoff_date}. "
            f"Need at least {min_training_rows:,}; increase --training-months or check raw data."
        )

    if max_train_rows is not None:
        if total > max_train_rows:
            fraction = max_train_rows / total
            train_df = train_df.sample(False, fraction, seed=random_seed).limit(max_train_rows)

    pdf = train_df.toPandas()
    if len(pdf) < min_training_rows:
        raise ValueError(
            f"Training data is too small after sampling: {len(pdf):,} rows. "
            f"Need at least {min_training_rows:,}; raise --max-train-rows or increase --training-months."
        )
    return pdf


def _baseline_stats(pdf: pd.DataFrame, anomaly_scores: np.ndarray) -> dict:
    feature_stats = {}
    for column in FEATURE_COLUMNS:
        values = pd.to_numeric(pdf[column], errors="coerce").fillna(0.0)
        feature_stats[column] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0) or 1.0),
            "p05": float(values.quantile(0.05)),
            "p50": float(values.quantile(0.50)),
            "p95": float(values.quantile(0.95)),
        }

    return {
        "feature_columns": FEATURE_COLUMNS,
        "feature_stats": feature_stats,
        "score_stats": {
            "mean": float(np.mean(anomaly_scores)),
            "std": float(np.std(anomaly_scores) or 1.0),
            "p50": float(np.quantile(anomaly_scores, 0.50)),
            "p95": float(np.quantile(anomaly_scores, 0.95)),
            "p99": float(np.quantile(anomaly_scores, 0.99)),
        },
    }


def train_isolation_forest(
    features: DataFrame,
    model_dir: str | Path,
    training_months: int = 12,
    contamination: float = CONTAMINATION,
    max_train_rows: int | None = None,
    random_seed: int = 42,
    min_training_rows: int = MIN_TRAINING_ROWS,
) -> dict:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    cutoff_date = training_cutoff_date(features, training_months, min_training_rows)
    train_pdf = _collect_training_frame(
        features, cutoff_date, max_train_rows, random_seed, min_training_rows
    )
    LOGGER.info(f"training rows used for model fit: {len(train_pdf):,}")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                IsolationForest(
                    n_estimators=200,
                    contamination=contamination,
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(train_pdf[FEATURE_COLUMNS])

    raw_scores = -pipeline.score_samples(train_pdf[FEATURE_COLUMNS])
    lower = float(np.quantile(raw_scores, 0.01))
    upper = float(np.quantile(raw_scores, 0.99))
    anomaly_scores = np.clip((raw_scores - lower) / max(upper - lower, 1e-9), 0.0, 1.0)

    metadata = _baseline_stats(train_pdf, anomaly_scores)
    metadata.update(
        {
            "training_cutoff_date": cutoff_date,
            "training_months": training_months,
            "contamination": contamination,
            "raw_score_p01": lower,
            "raw_score_p99": upper,
            "max_train_rows": max_train_rows,
            "training_rows": int(len(train_pdf)),
        }
    )

    joblib.dump(pipeline, model_path / MODEL_FILE)
    with (model_path / METADATA_FILE).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


