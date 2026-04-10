import asyncio
import os
import re
from openai import OpenAI
from client import OrangePigeonEnv
from models import OrangePigeonAction

async def main() -> None:
    # 1. EXACTLY AS REQUESTED BY SCALER GRADER FOR PROXY
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    openenv_url = os.environ.get("OPENENV_URL")
    image_name = os.environ.get("LOCAL_IMAGE_NAME")
    task_name = "orange_pigeon_defense"

    # 2. MANDATORY LOGGING: START
    print(f"[START] task={task_name} env=orange_pigeon_v1 model={model_name}", flush=True)

    env = None
    steps_taken = 0
    rewards = []

    try:
        if openenv_url:
            env = await OrangePigeonEnv.from_url(openenv_url)
        elif image_name:
            env = await OrangePigeonEnv.from_docker_image(image_name)
        else:
            env = await OrangePigeonEnv.from_env("aviralsach/orange-pigeon")

        result = await env.reset()
        last_state = result.observation.state

        for step in range(1, 11):
            if result.observation.done: break
            steps_taken = step

            # LLM Proxy Call
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"State: {last_state}. Output 1 integer (0,1,2)."}],
                max_tokens=10
            )
            
            action_text = completion.choices[0].message.content.strip()
            match = re.search(r'\d+', action_text)
            action_int = int(match.group(0)) if match else 0

            # Step in env
            result = await env.step(OrangePigeonAction(action=action_int))
            reward = result.observation.reward or 0.0
            rewards.append(reward)
            last_state = result.observation.state
            done_val = str(result.observation.done).lower()
            
            # MANDATORY LOGGING: STEP
            print(f"[STEP] step={step} action={action_int} reward={reward:.2f} done={done_val} error=null", flush=True)
            
    except Exception as e:
        print(f"[DEBUG] Error: {e}", flush=True)
        
    finally:
        if env: await env.close()
        
        # MANDATORY LOGGING: END
        score = sum(rewards) / 10.0 if rewards else 0.0
        success_val = str(score >= 0.5).lower()
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        print(f"[END] success={success_val} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())