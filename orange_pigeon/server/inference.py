import asyncio
import os
import re

from openai import AsyncOpenAI
from client import OrangePigeonEnv
from models import OrangePigeonAction

# ---------------------------------------------------------------------------
# Configuration - EXACT SYNTAX FOR SCALER GRADER (DO NOT CHANGE)
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-72B-Instruct")

async def main() -> None:
    # Initialize Async Client exactly as requested
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    task_name = "orange_pigeon_defense"
    print(f"[START] task={task_name} env=orange_pigeon_v1 model={MODEL}", flush=True)

    openenv_url = os.environ.get("OPENENV_URL")
    image_name = os.environ.get("LOCAL_IMAGE_NAME")
    
    env_manager = None

    try:
        if openenv_url:
            env_manager = OrangePigeonEnv(base_url=openenv_url)
        elif image_name:
            env_manager = await OrangePigeonEnv.from_docker_image(image_name)
        else:
            env_manager = await OrangePigeonEnv.from_env("aviralsach/orange-pigeon")

        async with env_manager as env:
            result = await env.reset()
            last_state = result.observation.state
            steps_taken = 0
            rewards = []

            for step in range(1, 11):
                if result.observation.done:
                    break
                steps_taken = step

                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": f"State: {last_state}. Output 1 integer (0,1,2)."}],
                    max_tokens=10,
                    temperature=0.0,
                )

                action_text = response.choices[0].message.content.strip()
                match = re.search(r'\d+', action_text)
                action_int = int(match.group(0)) if match else 0
                action_int = min(2, max(0, action_int))

                result = await env.step(OrangePigeonAction(action=action_int))
                reward = result.observation.reward or 0.0
                rewards.append(reward)
                last_state = result.observation.state

                print(
                    f"[STEP] step={step} action={action_int} reward={reward:.2f} "
                    f"done={str(result.observation.done).lower()} error=null",
                    flush=True,
                )

            score = sum(rewards) / 10.0 if rewards else 0.0
            success_val = str(score >= 0.5).lower()
            rewards_str = ",".join([f"{r:.2f}" for r in rewards])
            print(
                f"[END] success={success_val} steps={steps_taken} score={score:.3f} rewards={rewards_str}",
                flush=True,
            )

    except Exception as e:
        print(f"Error during execution: {e}", flush=True)
        print(f"[END] success=false steps=0 score=0.0 rewards=", flush=True)

if __name__ == "__main__":
    asyncio.run(main())