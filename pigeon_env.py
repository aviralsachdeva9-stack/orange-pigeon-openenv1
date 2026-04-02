import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OrangePigeonEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # ACTIONS: Agent kya kar sakta hai?
        # 0 = Do nothing, 1 = Low Sound, 2 = High Frequency Deterrent
        self.action_space = spaces.Discrete(3)
        
        # STATE: Agent kya dekh raha hai?
        # Man lo 2 cheezein: [Pigeon Present (0 or 1), Current Noise Level (0 to 10)]
        self.observation_space = spaces.MultiDiscrete([2, 11])
        
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start of the simulation: Pigeon aa gaya (1), noise level 0 hai
        self.state = np.array([1, 0])
        return self.state, {}

    def step(self, action):
        # YAHAN HUMARA MAIN LOGIC AAYEGA
        # 1. Agent ne jo action liya, uska asar calculate karenge
        # 2. Reward calculate karenge
        # 3. Check karenge ki kya task 'done' ho gaya
        
        reward = 0
        terminated = False
        
        # (Next step mein hum is logic ko fill karenge)
        
        return self.state, reward, terminated, False, {}

# Test karne ke liye chota sa block
if __name__ == "__main__":
    env = OrangePigeonEnv()
    initial_state, _ = env.reset()
    print(f"Environment Initialized! Starting State: {initial_state}")