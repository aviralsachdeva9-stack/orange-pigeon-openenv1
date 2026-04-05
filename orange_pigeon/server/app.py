from openenv.core.env_server import create_app, Environment
from models import OrangePigeonAction, OrangePigeonObservation, OrangePigeonState
from client import OrangePigeonEnv
import numpy as np

class HostedEnv(Environment):
    def __init__(self, **kwargs):
        super().__init__()  
        self.internal_env = OrangePigeonEnv(**kwargs)
        self._last_reward = 0.0
        self._last_done = False

    @property
    def state(self) -> OrangePigeonState:
        """Return the current state as OrangePigeonState"""
        return OrangePigeonState(
            episode_id="default",
            step_count=self.internal_env.current_step
        )

    def reset(self, *args, **kwargs) -> OrangePigeonObservation:
        """Reset the environment and return initial observation"""
        super().reset()  
        obs, info = self.internal_env.reset()
        obs_list = obs.tolist() if isinstance(obs, np.ndarray) else list(obs)
        self._last_reward = 0.0
        self._last_done = False
        return OrangePigeonObservation(
            state=obs_list,
            reward=0.0,
            done=False,
            step_count=0
        )
        
    def step(self, action: OrangePigeonAction, *args, **kwargs) -> OrangePigeonObservation:
        """Step the environment and return observation"""
        super().step(action)  
        
        act_val = action.action
        obs, reward, terminated, truncated, info = self.internal_env.step(act_val)
        obs_list = obs.tolist() if isinstance(obs, np.ndarray) else list(obs)
        
        self._last_reward = reward
        self._last_done = terminated or truncated
        
        return OrangePigeonObservation(
            state=obs_list,
            reward=float(reward),
            done=terminated or truncated,
            step_count=self.internal_env.current_step
        )

# THE ONLY FIX: Wapas HostedEnv class pass karni hai bina brackets ke!
app = create_app(HostedEnv, OrangePigeonAction, OrangePigeonObservation)
# --- YEH LINES SABSE NEECHE ADD KARNI HAIN ---
import uvicorn

# --- File ke end mein ---
def main():
    # Port 8000 ko 7860 mein change karna hai
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()