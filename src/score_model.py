from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from src.config import SUSPICIOUS_REVIEW_THRESHOLD
from src.features import FEATURE_COLUMNS
from src.train_model import METADATA_FILE, MODEL_FILE


def _load_metadata(model_dir: str | Path) -> dict:
    with (Path(model_dir) / METADATA_FILE).open("r", encoding="utf-8") as f:
        return json.load(f)


def score_reviews(features: DataFrame, model_dir: str | Path) -> DataFrame:
    spark = features.sparkSession
    model = joblib.load(Path(model_dir) / MODEL_FILE)
    metadata = _load_metadata(model_dir)
    model_bc = spark.sparkContext.broadcast(model)
    lower = float(metadata["raw_score_p01"])
    upper = float(metadata["raw_score_p99"])
    cutoff_date = metadata["training_cutoff_date"]

    @F.pandas_udf(DoubleType())
    def anomaly_score_udf(*cols: pd.Series) -> pd.Series:
        pdf = pd.concat(cols, axis=1)
        pdf.columns = FEATURE_COLUMNS
        raw_scores = -model_bc.value.score_samples(pdf[FEATURE_COLUMNS])
        normalized = np.clip((raw_scores - lower) / max(upper - lower, 1e-9), 0.0, 1.0)
        return pd.Series(normalized.astype(float))

    return (
        features.where(F.col("review_month") >= F.lit(cutoff_date))
        .withColumn("anomaly_score", anomaly_score_udf(*[F.col(c) for c in FEATURE_COLUMNS]))
        .withColumn(
            "is_suspicious_review",
            (F.col("anomaly_score") >= F.lit(SUSPICIOUS_REVIEW_THRESHOLD)).cast("double"),
        )
    )
