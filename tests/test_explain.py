from src.explain import explain_business_month


def test_explain_mentions_high_anomaly_signal():
    explanation = explain_business_month(
        {
            "business_name": "Test Cafe",
            "review_month": "2022-01-01",
            "max_anomaly_score": 0.91,
            "suspicious_review_density": 0.10,
        }
    )

    assert "Test Cafe" in explanation
    assert "anomaly" in explanation
