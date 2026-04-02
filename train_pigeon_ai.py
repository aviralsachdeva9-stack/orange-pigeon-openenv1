import gymnasium as gym
from stable_baselines3 import PPO
from pigeon_env import OrangePigeonEnv
import os

# 1. Environment load karo
env = OrangePigeonEnv()

# 2. PPO Model initialize karo (Meta's favorite algorithm)
# 'MlpPolicy' matlab Neural Network ka use ho raha hai
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")

print("🚀 Training Shuru Ho Rahi Hai... AI kabootar bhagana seekh raha hai.")

# 3. AI ko 20,000 baar practice karne do
model.learn(total_timesteps=20000)

# 4. Trained Model ko save karo
model.save("orange_pigeon_smart_model")
print("✅ Training Complete! Model save ho gaya.")

# 5. Test karo ki AI ab kitna smart hai
print("\n--- Smart AI Testing ---")
obs, _ = env.reset()
for i in range(10):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info, _ = env.step(action)
    action_name = ["Nothing", "Low Sound", "High Sound"][action]
    print(f"Step {i+1}: AI chose {action_name} | State: {obs} | Reward: {reward}")
    if done: break