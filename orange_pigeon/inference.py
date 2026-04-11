import asyncio
import os
import re
from openai import AsyncOpenAI
from client import OrangePigeonEnv
from models import OrangePigeonAction

# Use the exact environment variable names provided
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4")
HF_TOKEN = os.getenv("HF_TOKEN")

async def main() -> None:
    if not HF_TOKEN:
        raise SystemExit("HF_TOKEN must be set to query the model.")

    print(f"[CONFIG] base_url={API_BASE_URL} model={MODEL_NAME}", flush=True)

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    task_name = "orange_pigeon_defense"
    print(f"[START] task={task_name} env=orange_pigeon_v1 model={MODEL_NAME}", flush=True)

    openenv_url = os.getenv("OPENENV_URL")
    image_name = os.getenv("LOCAL_IMAGE_NAME")
    env = None
    steps_taken = 0
    rewards = []

    try:
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
                print(f"[API_CALL] step={step} sending request to proxy...", flush=True)

                response = await client.chat.completions.create(
                    model=MODEL_NAME,
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