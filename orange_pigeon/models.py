from openenv.core.env_server import Action, Observation, State
from typing import List
from pydantic import BaseModel

class OrangePigeonAction(Action):
    action: int

class OrangePigeonObservation(Observation):
    state: List[int]
    reward: float = 0.0
    done: bool = False
    step_count: int = 0

class OrangePigeonState(State):
    episode_id: str = "default"
    step_count: int = 0