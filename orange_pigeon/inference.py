import asyncio
import os
import textwrap
import re
from typing import List, Optional

from openai import OpenAI

# Tumhari files se Client aur Action models import kar rahe hain
from client import OrangePigeonEnv
from models import OrangePigeonAction

# Mandatory Variables for Hackathon
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

# IMPORTANT: Added this variable for the Hackathon auto-grader
OPENENV_URL = os.getenv("OPENENV_URL")

TASK_NAME = "orange_pigeon_defense"
BENCHMARK = "orange_pigeon_v1"
MAX_STEPS = 10
TEMPERATURE = 0.3 
MAX_TOKENS = 50
SUCCESS_SCORE_THRESHOLD = 0.5  

# System Prompt LLM ke dimaag ke liye
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Agent controlling an 'Orange Pigeon' defense system.
    At each step, you will receive the 'Current State' of the environment as a list of numbers.
    Your goal is to choose the correct action to maximize the reward.
    Valid actions are single integers (e.g., 0, 1, or 2).
    Reply with exactly ONE INTEGER. Do not add any text, explanations, or punctuation.
    """
).strip()

# --- STRICT LOGGING FUNCTIONS FOR AUTO-GRADER ---

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- AGENT LOGIC ---

def build_user_prompt(step: int, last_state: list, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current State: {last_state}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        
        Determine the next action. Output ONLY a single integer.
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, last_state: list, last_reward: float, history: List[str]) -> int:
    user_prompt = build_user_prompt(step, last_state, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Regex to safely extract the first integer LLM spits out
        match = re.search(r'\d+', text)
        return int(match.group(0)) if match else 0
        
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 0 # Fallback action

# --- MAIN LOOP ---

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    env = None # THE FIX: Initialize env variable here to prevent UnboundLocalError

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # THE FIX: Moved Network Connection INSIDE the try block!
        if OPENENV_URL:
            # Note: Check your client.py if the exact method name differs from from_url
            env = await OrangePigeonEnv.from_url(OPENENV_URL) 
        elif IMAGE_NAME:
            env = await OrangePigeonEnv.from_docker_image(IMAGE_NAME)
        else:
            env = await OrangePigeonEnv.from_env("aviralsach/orange-pigeon")

        result = await env.reset()
        last_state = result.observation.state
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.observation.done:
                break

            # LLM decides the action
            action_int = get_model_action(client, step, last_state, last_reward, history)

            # Take step in environment
            result = await env.step(OrangePigeonAction(action=action_int))
            obs = result.observation

            reward = obs.reward or 0.0
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_state = obs.state
            last_reward = reward

            log_step(step=step, action=str(action_int), reward=reward, done=done, error=error)
            history.append(f"Step {step}: Action {action_int} -> Reward {reward:+.2f}")

            if done:
                break

        # Calculate final normalized score
        max_possible_reward = float(MAX_STEPS) # Adjust based on your max possible reward logic
        score = sum(rewards) / max_possible_reward if max_possible_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        # THE FIX: Grader won't crash now, it will safely print this and move to log_end
        print(f"[DEBUG] Runtime/Connection Error: {e}", flush=True)

    finally:
        # THE FIX: Ensure we only close if env was successfully created
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)
            
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())