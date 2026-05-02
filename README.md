# Marketplace Integrity Monitor

A local, production-style data science portfolio project for detecting suspicious marketplace product-review behavior with PySpark, anomaly detection, drift monitoring, and a human-in-the-loop moderation queue.

The active dataset is **Amazon Reviews 2023**, currently the `All_Beauty` category. The reviews are historical even though the dataset release is named 2023. This project replays those historical timestamps as if reviews were arriving over time, which allows behavioral drift, model drift, and emerging suspicious review clusters to be evaluated month by month.

No cloud services, Kafka, or external APIs are required for the pipeline. The optional Streamlit LLM explanation uses `OPENAI_API_KEY` only if you choose to configure it.

## Why this project exists

This project is designed to fill common senior and lead data science portfolio gaps in one local, runnable system:

- PySpark / distributed processing for local batch ingestion and feature engineering.
- Anomaly detection with an Isolation Forest model.
- Behavioral drift monitoring at the product-month decision level.
- Model drift monitoring for feature and prediction distribution shifts.
- Human-in-the-loop review through a local moderation queue and Streamlit labeling app.
- Production ML lifecycle habits: modular code, persisted datasets, model metadata, monitoring outputs, tests, and clear operating instructions.

## Dataset choice

Amazon Reviews 2023: `All_Beauty` is the active category because product-review manipulation is common in marketplace settings and can include short 5-star reviews, repeated or templated language, suspicious review bursts, and review reuse across listings.

This local version is intentionally locked to `All_Beauty` to keep the project small and runnable.

The system replays historical review timestamps as if new reviews arrive daily. This lets the pipeline measure behavioral drift, model drift, and emerging suspicious review clusters over time while staying local and portfolio-sized.

## Anomaly Detection vs Reviewer Prioritization

Anomaly detection finds statistically unusual reviews. Not all anomalies are fraud. The reviewer-facing `fraud_like_score` adds fraud-oriented rules so moderators see cases that are more likely to require investigation, such as very short high-rating reviews, large rating deviations, near-duplicate text, formatting anomalies, reviewer behavior patterns, review bursts, clusters of extreme ratings, and unverified high-rating reviews when purchase verification is available.

Long detailed reviews and high helpful-vote counts are de-emphasized when they are not paired with stronger suspicious patterns.

Includes an experimental synthetic/templated language signal for generic or possibly AI-like review text; this is treated as a weak prioritization feature, not a classifier.

The queue is intentionally tuned for precision over recall, surfacing fewer but stronger candidates for review.

## What It Builds

- Amazon Reviews 2023 ingestion adapter for local category files.
- Normalized review schema using existing internal pipeline fields.
- Review-level feature engineering:
  - rating, text length, word count, helpful votes
  - rating deviation from product average
  - reviewer average rating, review count, and extreme-rating share
  - lightweight bot-like text signals such as repeated characters, generic praise, HTML breaks, and formatting anomalies
- Review-level anomaly scoring with an Isolation Forest trained on an early historical window.
- Product-month aggregation for operational moderation decisions.
- Behavioral drift metrics:
  - review volume change
  - rating shift
  - extreme rating share shift
  - review length shift
- Model monitoring outputs:
  - feature drift versus training baseline
  - prediction drift from anomaly score distribution
  - model drift score
  - retraining alert
- Moderation queue with reviewer-facing explanations.
- Streamlit review app for flagged product-months and local moderator labels.

## Project Structure

```text
marketplace-integrity-monitor/
|-- app/
|   `-- streamlit_app.py
|-- data/
|   |-- raw/
|   |   `-- amazon/
|   |-- processed/
|   `-- labels/
|-- outputs/
|   |-- moderation_queue/
|   |-- model_monitoring/
|   `-- drift_metrics/
|-- src/
|   |-- config.py
|   |-- ingest_amazon.py
|   |-- features.py
|   |-- train_model.py
|   |-- score_model.py
|   |-- behavioral_drift.py
|   |-- model_drift.py
|   |-- decisions.py
|   |-- explain.py
|   |-- bandit.py
|   `-- run_pipeline.py
|-- tests/
|-- README.md
|-- requirements.txt
|-- Dockerfile
`-- docker-compose.yml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the Amazon Reviews 2023 category file under `data/raw/amazon`, for example:

```text
data/raw/amazon/All_Beauty.jsonl.gz
```

Optional product metadata can be placed alongside it if available:

```text
data/raw/amazon/meta_All_Beauty.jsonl.gz
```

When metadata is present, the Streamlit app uses product titles. When metadata is missing, it displays `Unknown product (ASIN ...)`.

## Run The Pipeline

From the project root:

```bash
python -m src.run_pipeline --max-train-rows 25000
```

The defaults in `src/config.py` are:

```python
DATASET = "amazon"
AMAZON_CATEGORY = "All_Beauty"
```


`--max-train-rows` caps only the local Isolation Forest training sample collected for scikit-learn. PySpark still performs the ingestion, feature engineering, scoring, aggregation, and monitoring steps over the loaded local category data.

## Docker / Linux Container Run

For the most stable local run, especially on Windows, use Docker from the project root:

```bash
docker compose run --rm pipeline
```

The compose setup mounts local folders into the container:

- `data/raw` is read-only input for Amazon category files.
- `data/processed`, `outputs`, and `models` are written back to your local project folder.

## Outputs

Outputs are local CSV files written through Pandas to avoid Windows Hadoop write issues:

```text
data/processed/review_features/data.csv
data/processed/scored_reviews/data.csv
outputs/drift_metrics/behavioral_drift/data.csv
outputs/model_monitoring/model_drift/data.csv
outputs/moderation_queue/[product-month queue]/data.csv
outputs/moderation_queue/top_suspicious_reviews/data.csv
models/isolation_forest.joblib
models/model_metadata.json
```

## Streamlit Review App

After running the pipeline:

```bash
streamlit run app/streamlit_app.py
```

The app shows flagged product-months, top suspicious reviews, reviewer-facing explanations, technical signal definitions, and per-review moderator labels.

Moderator labels are appended locally to:

```text
data/labels/moderator_labels.csv
```

Labels are saved locally and can be used to evaluate precision, review false positives, and tune thresholds. In this prototype, labels do not automatically retrain the Isolation Forest model.

Future improvement: use moderator labels for active learning, supervised calibration, or threshold tuning once enough reviewed examples are collected.

## Configurable Thresholds

Defaults live in `src/config.py`:

- `SUSPICIOUS_REVIEW_THRESHOLD`: review-level anomaly score cutoff for suspicious review density.
- `DECISION_THRESHOLD`: product-month cutoff for routing to the moderation queue.
- `RETRAINING_ALERT_THRESHOLD`: model drift score cutoff for retraining alerts.
- `CONTAMINATION`: Isolation Forest contamination assumption.
- `MIN_TRAINING_ROWS`: minimum rows required before model training proceeds.
- `DATASET`: active dataset, currently `amazon`.
- `AMAZON_CATEGORY`: active Amazon category, currently `All_Beauty`.

## Troubleshooting

Native Windows Spark can fail during Parquet writes because the bundled Hadoop libraries and local `NativeIO` behavior do not always match the installed Windows environment. This project writes final outputs through Pandas CSV to avoid Hadoop output commits on Windows.

For a more production-like Spark runtime, run the pipeline in WSL2 or Docker.

## Tests

```bash
pytest
```



