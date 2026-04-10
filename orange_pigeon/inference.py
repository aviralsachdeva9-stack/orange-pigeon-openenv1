import asyncio
import os
import textwrap
import re
from typing import List, Optional
from openai import OpenAI
from client import OrangePigeonEnv
from models import OrangePigeonAction

# Constants
TASK_NAME = "orange_pigeon_defense"
BENCHMARK = "orange_pigeon_v1"
MAX_STEPS = 10

async def main() -> None:
    # --- CRITICAL FIX FOR SCALER GRADER ---
    # Inhone kaha hai exactly 'os.environ["..."]' use karo, toh hum wahi kar rahe hain
    try:
        api_base_url = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]
        client = OpenAI(base_url=api_base_url, api_key=api_key)
    except KeyError:
        # Local testing ke liye fallback
        api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        api_key = os.getenv("HF_TOKEN", "dummy-key")
        client = OpenAI(base_url=api_base_url, api_key=api_key)

    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    openenv_url = os.getenv("OPENENV_URL")
    image_name = os.getenv("LOCAL_IMAGE_NAME")

    # Logging start
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={model_name}", flush=True)

    env = None
    try:
        if openenv_url:
            env = await OrangePigeonEnv.from_url(openenv_url)
        elif image_name:
            env = await OrangePigeonEnv.from_docker_image(image_name)
        else:
            env = await OrangePigeonEnv.from_env("aviralsach/orange-pigeon")

        result = await env.reset()
        last_state = result.observation.state
        rewards = []

        for step in range(1, MAX_STEPS + 1):
            if result.observation.done: break

            # LLM call through the PROXY
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"State: {last_state}. Output 1 integer (0,1,2)."}],
                max_tokens=10
            )
            
            action_text = completion.choices[0].message.content.strip()
            action_int = int(re.search(r'\d+', action_text).group(0)) if re.search(r'\d+', action_text) else 0

            result = await env.step(OrangePigeonAction(action=action_int))
            reward = result.observation.reward or 0.0
            rewards.append(reward)

            print(f"[STEP] step={step} action={action_int} reward={reward:.2f} done={result.observation.done}", flush=True)
            if result.observation.done: break

        score = sum(rewards) / MAX_STEPS
        print(f"[END] success={str(score >= 0.5).lower()} steps={len(rewards)} score={score:.3f} rewards={','.join(map(str, rewards))}", flush=True)

    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)
    finally:
        if env: await env.close()

if __name__ == "__main__":
    asyncio.run(main())