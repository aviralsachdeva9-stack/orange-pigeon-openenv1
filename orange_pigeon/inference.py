import asyncio
import os
import re
from openai import AsyncOpenAI  # Must be Async for use in async context
from client import OrangePigeonEnv
from models import OrangePigeonAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

MODEL = os.getenv("MODEL", "Qwen/Qwen2.5-72B-Instruct")

# ---------------------------------------------------------------------------
# Gameplay Logic
# ---------------------------------------------------------------------------
async def main() -> None:
    if not API_KEY:
        raise SystemExit("API_KEY (or HF_TOKEN) must be set to query the model.")

    # Use AsyncOpenAI — regular OpenAI client blocks the event loop
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_name = "orange_pigeon_defense"
    print(f"[START] task={task_name} env=orange_pigeon_v1 model={MODEL}", flush=True)

    openenv_url = os.getenv("OPENENV_URL")
    image_name = os.getenv("LOCAL_IMAGE_NAME")
    env = None
    steps_taken = 0
    rewards = []

    try:
        # OrangePigeonEnv must be used as an async context manager
        if openenv_url:
            env = OrangePigeonEnv(base_url=openenv_url)
        elif image_name:
            env = await OrangePigeonEnv.from_docker_image(image_name)
        else:
            env = await OrangePigeonEnv.from_env("aviralsach/orange-pigeon")

        async with env:
            result = await env.reset()
            last_state = result.observation.state

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

                # Clamp to valid action range
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

    except Exception as e:
        print(f"Error during execution: {e}", flush=True)

    finally:
        score = sum(rewards) / 10.0 if rewards else 0.0
        success_val = str(score >= 0.5).lower()
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(
            f"[END] success={success_val} steps={steps_taken} score={score:.3f} rewards={rewards_str}",
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())