"""
Reinforcement Learning-based threshold optimization for FAQ retrieval.
"""

import json
import logging
import os
import random

logger = logging.getLogger(__name__)

THRESHOLD_STATE_FILE = "threshold_states.json"


class ThresholdOptimizer:
    """
    Multi-armed bandit approach for optimizing similarity thresholds.
    """

    def __init__(
        self,
        threshold_range: tuple[float, float] = (0.3, 0.9),
        num_arms: int = 10,
        epsilon: float = 0.1,
        decay_rate: float = 0.995,
    ):
        """
        Initialize the threshold optimizer.

        Args:
            threshold_range: Min and max threshold values to explore
            num_arms: Number of discrete threshold values to test
            epsilon: Exploration rate for epsilon-greedy strategy
            decay_rate: Rate at which epsilon decays over time
        """
        self.threshold_range = threshold_range
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.decay_rate = decay_rate

        # Create discrete threshold values
        self.thresholds = [threshold_range[0] + i * (threshold_range[1] - threshold_range[0]) / (num_arms - 1)
                          for i in range(num_arms)]

        # Initialize Q-values and counts
        self.q_values = [0.0] * num_arms
        self.action_counts = [0] * num_arms
        self.total_steps = 0

        # Load previous state if exists
        self._load_state()

        # Create initial state file if it doesn't exist
        if not os.path.exists(THRESHOLD_STATE_FILE):
            self._save_state()

        logger.info("Initialized ThresholdOptimizer with %d arms", num_arms)

    def select_threshold(self) -> float:
        """
        Select threshold using epsilon-greedy strategy.

        Returns:
            Selected threshold value
        """
        self.total_steps += 1

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Explore: random selection
            action = random.randint(0, self.num_arms - 1)
            logger.debug("Exploring with threshold: %.3f", self.thresholds[action])
        else:
            # Exploit: select best known action
            action = self.q_values.index(max(self.q_values))
            logger.debug("Exploiting with threshold: %.3f", self.thresholds[action])

        # Track action selection for analytics (increment usage count)
        self.action_counts[action] += 1

        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.decay_rate)

        # Save state periodically
        if self.total_steps % 5 == 0:
            self._save_state()

        return self.thresholds[action]

    def update_reward(self, threshold: float, reward: float) -> None:
        """
        Update Q-value based on received reward.

        Args:
            threshold: The threshold that was used
            reward: Reward received (1.0 for positive feedback, 0.0 for negative)
        """
        # Find the arm index for this threshold
        action = self._threshold_to_action(threshold)

        if action is not None:
            # Count how many times this action received feedback (not just selections)
            feedback_count = sum(1 for i in range(len(self.q_values)) if self.q_values[i] != 0.0)

            # Update Q-value using incremental average
            # Use the current number of times this specific action got feedback
            current_feedback_count = 1 if self.q_values[action] == 0.0 else feedback_count
            step_size = 1.0 / max(current_feedback_count, 1)
            self.q_values[action] += step_size * (reward - self.q_values[action])

            logger.debug("Updated Q-value for threshold %.3f: %.3f", threshold, self.q_values[action])

            # Save state after each update
            self._save_state()

    def get_best_threshold(self) -> float:
        """
        Get the currently best performing threshold.

        Returns:
            Best threshold value
        """
        best_action = self.q_values.index(max(self.q_values))
        return self.thresholds[best_action]

    def get_performance_stats(self) -> dict:
        """
        Get performance statistics for analysis.

        Returns:
            Dictionary with performance metrics
        """
        return {
            "best_threshold": float(self.get_best_threshold()),
            "best_q_value": float(max(self.q_values)),
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "q_values": self.q_values,
            "thresholds": self.thresholds,
            "action_counts": self.action_counts,
        }

    def _threshold_to_action(self, threshold: float) -> int | None:
        """
        Convert threshold value to action index.

        Args:
            threshold: Threshold value

        Returns:
            Action index or None if not found
        """
        # Find closest threshold
        distances = [abs(t - threshold) for t in self.thresholds]
        action = distances.index(min(distances))

        # Only return if reasonably close
        if distances[action] < 0.05:
            return action
        return None

    def _save_state(self) -> None:
        """
        Save optimizer state to file.
        """
        try:
            state = {
                "q_values": self.q_values,
                "action_counts": self.action_counts,
                "total_steps": self.total_steps,
                "epsilon": self.epsilon,
                "thresholds": self.thresholds,
            }

            with open(THRESHOLD_STATE_FILE, "w") as f:
                json.dump(state, f)

            logger.debug("Saved optimizer state")

        except Exception as e:
            logger.error("Failed to save optimizer state: %s", str(e))

    def _load_state(self) -> None:
        """
        Load optimizer state from file.
        """
        try:
            if os.path.exists(THRESHOLD_STATE_FILE):
                with open(THRESHOLD_STATE_FILE) as f:
                    state = json.load(f)

                self.q_values = state.get("q_values", self.q_values)
                self.action_counts = state.get("action_counts", self.action_counts)
                self.total_steps = state.get("total_steps", 0)
                self.epsilon = state.get("epsilon", self.initial_epsilon)

                logger.info("Loaded optimizer state from previous session")

        except Exception as e:
            logger.error("Failed to load optimizer state: %s", str(e))

    def reset(self) -> None:
        """
        Reset the optimizer to initial state.
        """
        self.q_values = [0.0] * self.num_arms
        self.action_counts = [0] * self.num_arms
        self.total_steps = 0
        self.epsilon = self.initial_epsilon

        # Remove state file
        if os.path.exists(THRESHOLD_STATE_FILE):
            os.remove(THRESHOLD_STATE_FILE)

        logger.info("Reset optimizer to initial state")
