from __future__ import annotations


SUSPICIOUS_REVIEW_THRESHOLD = 0.70
DECISION_THRESHOLD = 0.70
RETRAINING_ALERT_THRESHOLD = 0.65
CONTAMINATION = 0.03
MIN_TRAINING_ROWS = 1_000

# Lowercase aliases make the interview-facing knobs match the README language.
suspicious_review_threshold = SUSPICIOUS_REVIEW_THRESHOLD
decision_threshold = DECISION_THRESHOLD
retraining_alert_threshold = RETRAINING_ALERT_THRESHOLD
contamination = CONTAMINATION

DATASET = "amazon"
AMAZON_CATEGORY = "All_Beauty"

# Lowercase aliases for dataset selection.
dataset = DATASET
amazon_category = AMAZON_CATEGORY


