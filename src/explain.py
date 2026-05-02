from __future__ import annotations


def explain_business_month(row: dict) -> str:
    reasons: list[str] = []

    if float(row.get("max_anomaly_score") or 0.0) >= 0.80:
        reasons.append("one or more reviews have unusually high anomaly scores")
    if float(row.get("suspicious_review_density") or 0.0) >= 0.25:
        reasons.append("a high share of reviews in this month are suspicious")
    if abs(float(row.get("review_volume_change") or 0.0)) >= 1.0:
        reasons.append("review volume changed sharply versus the prior month")
    if abs(float(row.get("rating_shift") or 0.0)) >= 1.0:
        reasons.append("average rating shifted materially versus the prior month")
    if abs(float(row.get("extreme_rating_share_shift") or 0.0)) >= 0.25:
        reasons.append("the share of extreme ratings changed materially")
    if float(row.get("model_drift_score") or 0.0) >= 0.65:
        reasons.append("model monitoring indicates drift in the scored population")

    if not reasons:
        reasons.append("combined anomaly and drift signals crossed the moderation threshold")

    business_name = row.get("business_name") or row.get("business_id") or "This business"
    month = row.get("review_month") or "the selected month"
    return f"{business_name} was flagged for {month} because " + "; ".join(reasons) + "."
