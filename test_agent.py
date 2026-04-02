import gymnasium as gym
from pigeon_env import OrangePigeonEnv

def run_random_agent(episodes=3):
    # Humara custom environment load karo
    env = OrangePigeonEnv()

    for ep in range(episodes):
        print(f"\n--- Episode {ep + 1} Start ---")
        state, _ = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            step_count += 1
            # AI randomly 0, 1, ya 2 choose kar raha hai
            action = env.action_space.sample() 
            
            # Action lo aur environment ka naya state dekho
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            # Action ka naam print karne ke liye
            action_name = ["Kuch Nahi Kiya", "Low Sound", "High Sound"][action]
            print(f"Step {step_count}: AI ne chuna '{action_name}' | Result: Pigeon={next_state[0]}, Noise={next_state[1]} | Reward: {reward}")

        print(f"Episode {ep + 1} Khatam! Total Score: {total_reward}")

if __name__ == "__main__":
    run_random_agent()