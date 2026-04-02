import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class OrangePigeonEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # ACTIONS: 0 = Kuch mat karo, 1 = Low Sound, 2 = High Frequency Deterrent
        self.action_space = spaces.Discrete(3)
        
        # STATE: [Pigeon Present (0 ya 1), Current Noise Level (0 se 10 tak)]
        self.observation_space = spaces.MultiDiscrete([2, 11])
        
        self.state = None
        self.max_steps = 20 # AI ko ek baari mein 20 steps (chances) milenge
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Start of Simulation: Pigeon aa gaya hai (1), aur aawaz bilkul nahi hai (0)
        self.state = np.array([1, 0])
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        pigeon_present = self.state[0]
        noise_level = self.state[1]

        reward = 0
        terminated = False

        # --- LOGIC 1: Action lene se Noise par kya asar padega ---
        if action == 0: # Do nothing
            noise_level = max(0, noise_level - 1) # Shanti badhegi (noise kam hoga)
        elif action == 1: # Low sound
            noise_level = min(10, noise_level + 2) # Thoda shor badhega
        elif action == 2: # High sound
            noise_level = min(10, noise_level + 5) # Bohot shor badhega

        # --- LOGIC 2: Pigeon udega ya nahi, aur Reward/Penalty kya hogi ---
        if pigeon_present == 1:
            if action == 0:
                reward = -2 # Penalty: Pigeon baitha hai aur tumne kuch nahi kiya!
            elif action == 1:
                # Low sound se 50% chance hai ki pigeon ud jayega
                if random.random() < 0.5:
                    pigeon_present = 0
                    reward = 10 # Jackpot! Pigeon ud gaya
                else:
                    reward = -1 # Pigeon dheet nikla, nahi uda
            elif action == 2:
                # High sound se 90% chance hai udne ka
                if random.random() < 0.9:
                    pigeon_present = 0
                    reward = 10 
                else:
                    reward = -1
        else:
            # Agar pigeon nahi hai aur phir bhi AI ne aawaz nikaali = VERY BAD
            if action > 0:
                reward = -5 # Noise pollution ki bhari penalty
            else:
                reward = 1 # Sab shant hai, aur koi kabootar nahi hai. Good boy!

        # --- LOGIC 3: Over-noise Penalty ---
        # Agar shor limit (7) se bahar gaya, toh extra penalty
        if noise_level > 7:
            reward -= 5

        # --- LOGIC 4: Episode kab khatam hoga? ---
        if pigeon_present == 0 and noise_level == 0:
            terminated = True # Success! Kabootar bhag gaya aur waapis shanti ho gayi
        elif self.current_step >= self.max_steps:
            terminated = True # Time khatam

        # State ko naye data ke sath update karo
        self.state = np.array([pigeon_present, noise_level])
        
        return self.state, reward, terminated, False, {}

# Test karne ke liye block
if __name__ == "__main__":
    env = OrangePigeonEnv()
    state, _ = env.reset()
    print(f"Start State: [Pigeon: {state[0]}, Noise: {state[1]}]")
    
    # AI ne try kiya Action 2 (High Sound)
    next_state, reward, done, _, _ = env.step(2)
    print(f"Action Taken: High Sound (2)")
    print(f"Next State: [Pigeon: {next_state[0]}, Noise: {next_state[1]}] | Reward: {reward} | Done: {done}")