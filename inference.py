#!/usr/bin/env python3
"""
Orange Pigeon Inference Script
Follows the official OpenENV format for Phase 2 validation
"""

import asyncio
import os
import re
import textwrap
from typing import List, Optional

from openai import AsyncOpenAI
from orange_pigeon.client import OrangePigeonEnv
from orange_pigeon.models import OrangePigeonAction

# Environment variables - MUST be injected by validator (no defaults)
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL")  # Required - validator's LiteLLM proxy URL
API_KEY = os.getenv("API_KEY")  # Required - validator's injected API key
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")  # Can have default

TASK_NAME = "orange_pigeon_defense"
BENCHMARK = "orange_pigeon_v1"
MAX_STEPS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 50

# Scoring: assuming max reward per step is ~10
_MAX_REWARD_PER_STEP = 10.0
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are controlling a pest deterrence system for an orange pigeon.
    Available actions:
    0: Do nothing (quietest)
    1: Play low sound
    2: Play high sound

    Goal: Scare away the pigeon while minimizing noise pollution.
    Reply with ONLY a single digit: 0, 1, or 2.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Log each step following official format"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log episode end with final metrics"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, state: list, last_reward: float, history: List[str]) -> str:
    """Build context-aware prompt for the model"""
    history_block = "\n".join(history[-3:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current state: Pigeon={state[0]}, Noise={state[1]}/10
        Last reward: {last_reward:.2f}
        Recent actions:
        {history_block}

        Choose action (0, 1, or 2):
        """
    ).strip()


async def get_model_action(
    client: AsyncOpenAI,
    step: int,
    state: list,
    last_reward: float,
    history: List[str]
) -> int:
    """Get action from LLM via API proxy"""
    user_prompt = build_user_prompt(step, state, last_reward, history)
    try:
        print(f"[DEBUG] Making API call to {API_BASE_URL} with model {MODEL_NAME}", flush=True)
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        print(f"[DEBUG] API call successful, got response", flush=True)
        text = (completion.choices[0].message.content or "").strip()
        # Extract first digit 0-2
        match = re.search(r'[0-2]', text)
        action = int(match.group(0)) if match else 0
        return action
    except Exception as exc:
        print(f"[ERROR] API call failed: {exc}", flush=True)
        raise  # Don't silently fail - propagate the error


async def main() -> None:
    """Main inference loop following official format"""
    # Validate required environment variables from validator
    print(f"[DEBUG] API_BASE_URL: {API_BASE_URL}", flush=True)
    print(f"[DEBUG] API_KEY: {'SET' if API_KEY else 'NOT SET'}", flush=True)
    print(f"[DEBUG] MODEL_NAME: {MODEL_NAME}", flush=True)

    if not API_BASE_URL:
        raise SystemExit("[ERROR] API_BASE_URL not set by validator! This must point to the LiteLLM proxy.")
    if not API_KEY:
        raise SystemExit("[ERROR] API_KEY not set by validator!")

    print(f"[DEBUG] Using validator-injected API_BASE_URL: {API_BASE_URL}", flush=True)
    print(f"[DEBUG] Initializing OpenAI client with proxy...", flush=True)

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Initialize environment
    print(f"[DEBUG] Initializing environment...", flush=True)
    env = None
    if IMAGE_NAME:
        print(f"[DEBUG] Using Docker image: {IMAGE_NAME}", flush=True)
        env = await OrangePigeonEnv.from_docker_image(IMAGE_NAME)
    else:
        print(f"[DEBUG] Using HF space: aviralsach/orange-pigeon", flush=True)
        env = await OrangePigeonEnv.from_env("aviralsach/orange-pigeon")

    print(f"[DEBUG] Environment initialized successfully", flush=True)

    # TEST: Make a direct API call to verify the proxy is working
    print(f"[DEBUG] === TESTING LITELLM PROXY CONNECTION ===", flush=True)
    try:
        print(f"[DEBUG] Making test API call to verify proxy...", flush=True)
        test_response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Say 'PROXY_WORKING'"}],
            max_tokens=10,
        )
        print(f"[DEBUG] TEST API CALL SUCCESSFUL! Response: {test_response.choices[0].message.content}", flush=True)
    except Exception as e:
        print(f"[ERROR] TEST API CALL FAILED: {e}", flush=True)
        raise
    print(f"[DEBUG] === LITELLM PROXY VERIFIED ===", flush=True)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        state = result.observation.state
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            print(f"[DEBUG] Step {step}: Checking if done={result.observation.done}", flush=True)
            if result.observation.done:
                print(f"[DEBUG] Environment is done, exiting loop", flush=True)
                break

            print(f"[DEBUG] Step {step}: About to call get_model_action() for LLM API call", flush=True)
            # Get action from LLM
            action_int = await get_model_action(client, step, state, last_reward, history)
            print(f"[DEBUG] Step {step}: Received action={action_int} from LLM", flush=True)
            action_str = ["do_nothing", "low_sound", "high_sound"][action_int]

            # Step environment
            result = await env.step(OrangePigeonAction(action=action_int))
            obs = result.observation

            reward = obs.reward or 0.0
            done = obs.done
            error = None

            rewards.append(reward)
            steps_taken = step
            state = obs.state
            last_reward = reward

            # Log step with exact format
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str} -> {reward:+.2f}")

            if done:
                break

        # Normalize score to [0, 1]
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= 0.5

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        # Always log end, even on exception
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
