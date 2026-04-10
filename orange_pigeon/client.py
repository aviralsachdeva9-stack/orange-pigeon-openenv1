import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class OrangePigeonEnv(gym.Env):
    def __init__(self, task_level=1):
        super().__init__()
        self.task_level = task_level
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.MultiDiscrete([2, 11])
        self.state = np.array([1, 0])
        self.max_steps = 20
        self.current_step = 0

        if self.task_level == 1:
            self.noise_threshold = 7
            self.stubbornness = 0.0
        elif self.task_level == 2:
            self.noise_threshold = 5
            self.stubbornness = 0.2
        else:
            self.noise_threshold = 3
            self.stubbornness = 0.4

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = np.array([1, 0])
        return self.state, {}

    def step(self, action):
        # Extremely safe action extraction
        if hasattr(action, 'action'):
            act_val = int(action.action)
        elif isinstance(action, dict):
            act_val = int(action.get('action', 0))
        else:
            act_val = int(action)

        self.current_step += 1
        pigeon_present = self.state[0]
        noise_level = self.state[1]
        reward = 0
        terminated = False

        if act_val == 0:
            noise_level = max(0, noise_level - 1)
        elif act_val == 1:
            noise_level = min(10, noise_level + 2)
        elif act_val == 2:
            noise_level = min(10, noise_level + 5)

        if pigeon_present == 1:
            if act_val == 0:
                reward = -2
            elif act_val == 1:
                if random.random() < (0.5 - self.stubbornness):
                    pigeon_present = 0
                    reward = 10
                else:
                    reward = -1
            elif act_val == 2:
                if random.random() < (0.9 - self.stubbornness):
                    pigeon_present = 0
                    reward = 10
                else:
                    reward = -1
        else:
            if act_val > 0:
                reward = -5
            else:
                reward = 1

        if noise_level > self.noise_threshold:
            reward -= 5

        if pigeon_present == 0 and noise_level == 0:
            terminated = True
        elif self.current_step >= self.max_steps:
            terminated = True

        self.state = np.array([pigeon_present, noise_level])
        return self.state, float(reward), terminated, False, {}

    def get_grader_score(self, total_reward):
        min_expected = -50.0
        max_expected = 12.0
        clipped_reward = max(min_expected, min(total_reward, max_expected))
        normalized_score = (clipped_reward - min_expected) / (max_expected - min_expected)
        return round(normalized_score, 2)