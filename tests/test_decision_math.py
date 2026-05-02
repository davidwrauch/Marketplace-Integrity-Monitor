def test_decision_score_weights_are_interpretable():
    score = 0.35 * 1.0 + 0.25 * 0.5 + 0.25 * 0.4 + 0.15 * 0.2
    assert round(score, 3) == 0.605
