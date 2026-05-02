from __future__ import annotations

import json
import random
from pathlib import Path


class EpsilonGreedyThresholdBandit:
    def __init__(
        self,
        thresholds: list[float] | None = None,
        epsilon: float = 0.10,
        state_path: str | Path = "outputs/moderation_queue/bandit_state.json",
    ) -> None:
        self.thresholds = thresholds or [0.60, 0.70, 0.80]
        self.epsilon = epsilon
        self.state_path = Path(state_path)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {str(t): {"trials": 0, "reward": 0.0} for t in self.thresholds}

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def choose_threshold(self) -> float:
        if random.random() < self.epsilon:
            return random.choice(self.thresholds)

        def average_reward(threshold: float) -> float:
            arm = self.state[str(threshold)]
            return arm["reward"] / max(arm["trials"], 1)

        return max(self.thresholds, key=average_reward)

    def update(self, threshold: float, label: str) -> None:
        reward = 1.0 if label == "suspicious" else 0.0
        key = str(threshold)
        if key not in self.state:
            self.state[key] = {"trials": 0, "reward": 0.0}
        self.state[key]["trials"] += 1
        self.state[key]["reward"] += reward
        self.save()
