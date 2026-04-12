import asyncio
import os
import re

from openai import AsyncOpenAI
from client import OrangePigeonEnv
from models import OrangePigeonAction

async def main() -> None:
    # 1. BEAT THE REGEX: EXACTLY ONE LINE (Do not break this into multiple lines)
    client = AsyncOpenAI(base_url=os.environ["API_BASE_URL"], api_key=os.environ["API_KEY"])

    # Model name flexibility (handles both setups safely)
    model_to_use = os.environ.get("MODEL_NAME", os.environ.get("MODEL", "gpt-4"))
    
    task_name = "orange_pigeon_defense"
    print(f"[START] task={task_name} env=orange_pigeon_v1 model={model_to_use}", flush=True)

    openenv_url = os.environ.get("OPENENV_URL")
    image_name = os.environ.get("LOCAL_IMAGE_NAME")
    
    env_manager = None
    if openenv_url:
        env_manager = OrangePigeonEnv(base_url=openenv_url)
    elif image_name:
        env_manager = await OrangePigeonEnv.from_docker_image(image_name)
    else:
        env_manager = await OrangePigeonEnv.from_env("aviralsach/orange-pigeon")

    # 2. NO SILENT FAILURES: Removed the outer try-except so we can see real errors if they happen!
    async with env_manager as env:
        result = await env.reset()
        last_state = result.observation.state
        steps_taken = 0
        rewards = []

        for step in range(1, 11):
            if result.observation.done:
                break
            
            steps_taken = step
            action_int = 0
            
            # 3. THE 3x RETRY LOOP (Force proxy connection)
            for attempt in range(3):
                try:
                    response = await client.chat.completions.create(
                        model=model_to_use,
                        messages=[{"role": "user", "content": f"State: {last_state}. Output 1 integer (0,1,2)."}],
                        max_tokens=10,
                        temperature=0.0,
                    )
                    action_text = response.choices[0].message.content.strip()
                    match = re.search(r'\d+', action_text)
                    if match:
                        action_int = int(match.group(0))
                    break  # Proxy call succeeded! Exit the retry loop.
                except Exception as api_error:
                    print(f"[API ERROR WARNING] Attempt {attempt+1} failed: {api_error}", flush=True)
                    if attempt == 2:
                        # If it fails 3 times, crash loudly! No more fake success!
                        raise
                    await asyncio.sleep(1)

            action_int = min(2, max(0, action_int))

            result = await env.step(OrangePigeonAction(action=action_int))
            reward = result.observation.reward or 0.0
            rewards.append(reward)
            last_state = result.observation.state

            print(f"[STEP] step={step} action={action_int} reward={reward:.2f} done={str(result.observation.done).lower()} error=null", flush=True)

        score = sum(rewards) / 10.0 if rewards else 0.0
        success_val = str(score >= 0.5).lower()
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={success_val} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())