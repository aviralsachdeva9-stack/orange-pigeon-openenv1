import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class OrangePigeonEnv(gym.Env):
    # Added 'task_level' to support the 3+ tasks requirement
    def __init__(self, task_level=1):
        super().__init__()
        self.task_level = task_level 
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete([2, 11])
        self.state = None
        self.max_steps = 20
        self.current_step = 0
        
        # Difficulty scaling based on Task Level
        if self.task_level == 1:
            self.noise_threshold = 7 # Easy: High noise tolerance
            self.stubbornness = 0.0  # Normal pigeon
        elif self.task_level == 2:
            self.noise_threshold = 5 # Medium: Stricter noise rules
            self.stubbornness = 0.2  # Pigeon is 20% harder to scare
        else:
            self.noise_threshold = 3 # Hard: Very strict noise rules
            self.stubbornness = 0.4  # Pigeon is 40% harder to scare

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([1, 0])
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        pigeon_present = self.state[0]
        noise_level = self.state[1]
        reward = 0
        terminated = False

        if action == 0:
            noise_level = max(0, noise_level - 1)
        elif action == 1:
            noise_level = min(10, noise_level + 2)
        elif action == 2:
            noise_level = min(10, noise_level + 5)

        if pigeon_present == 1:
            if action == 0:
                reward = -2
            elif action == 1:
                # Task difficulty affects probability
                if random.random() < (0.5 - self.stubbornness):
                    pigeon_present = 0
                    reward = 10
                else:
                    reward = -1
            elif action == 2:
                if random.random() < (0.9 - self.stubbornness):
                    pigeon_present = 0
                    reward = 10
                else:
                    reward = -1
        else:
            if action > 0:
                reward = -5
            else:
                reward = 1

        # Use dynamic noise threshold based on task level
        if noise_level > self.noise_threshold:
            reward -= 5

        if pigeon_present == 0 and noise_level == 0:
            terminated = True
        elif self.current_step >= self.max_steps:
            terminated = True

        self.state = np.array([pigeon_present, noise_level])
        return self.state, reward, terminated, False, {}

    # NEW: Required 0.0 to 1.0 Grader Score
    def get_grader_score(self, total_reward):
        """
        Normalizes the raw reward into a 0.0 to 1.0 scale for OpenEnv evaluation.
        Assuming max possible clean run is ~12, and worst is ~ -100.
        """
        min_expected = -50.0
        max_expected = 12.0
        
        # Clip the reward within expected bounds
        clipped_reward = max(min_expected, min(total_reward, max_expected))
        
        # Normalize to 0.0 - 1.0
        normalized_score = (clipped_reward - min_expected) / (max_expected - min_expected)
        return round(normalized_score, 2)