from src.bandit import EpsilonGreedyThresholdBandit


def test_bandit_updates_reward(tmp_path):
    bandit = EpsilonGreedyThresholdBandit(epsilon=0.0, state_path=tmp_path / "state.json")
    bandit.update(0.7, "suspicious")

    assert bandit.state["0.7"]["trials"] == 1
    assert bandit.state["0.7"]["reward"] == 1.0
