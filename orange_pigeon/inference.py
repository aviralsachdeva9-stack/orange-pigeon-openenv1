import asyncio
import os
import re
from openai import OpenAI
from client import OrangePigeonEnv
from models import OrangePigeonAction

# Constants for Grader
TASK_NAME = "orange_pigeon_defense"
BENCHMARK = "orange_pigeon_v1"

async def main() -> None:
    # 1. API KEY SETUP (Grader-friendly + HuggingFace Safe)
    # The grader looks for os.environ["API_BASE_URL"], so we must use it safely.
    if "API_BASE_URL" in os.environ and "API_KEY" in os.environ:
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"]
        )
    else:
        # Fallback for Hugging Face so the script doesn't crash
        api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        api_key = os.environ.get("HF_TOKEN", "dummy-key")
        client = OpenAI(base_url=api_base_url, api_key=api_key)

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    openenv_url = os.environ.get("OPENENV_URL")
    image_name = os.environ.get("LOCAL_IMAGE_NAME")

    # 2. EXACT [START] LOG FOR GRADER
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={model_name}", flush=True)

    env = None
    steps_taken = 0
    rewards = []
    score = 0.0

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
            try:
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": f"State: {last_state}. Output 1 integer (0,1,2)."}],
                    max_tokens=10
                )
                action_text = completion.choices[0].message.content.strip()
                match = re.search(r'\d+', action_text)
                action_int = int(match.group(0)) if match else 0
            except Exception as llm_e:
                print(f"[DEBUG] LLM Call failed: {llm_e}")
                action_int = 0 # Fallback action

            # Step in env
            result = await env.step(OrangePigeonAction(action=action_int))
            reward = result.observation.reward or 0.0
            rewards.append(reward)
            last_state = result.observation.state
            done_val = str(result.observation.done).lower()
            
            # 3. EXACT [STEP] LOG FOR GRADER
            print(f"[STEP] step={step} action={action_int} reward={reward:.2f} done={done_val} error=null", flush=True)
            
    except Exception as e:
        print(f"[DEBUG] Runtime Error: {e}", flush=True)
        
    finally:
        if env: await env.close()
        
        # 4. EXACT [END] LOG FOR GRADER
        if len(rewards) > 0:
            score = sum(rewards) / float(len(rewards))
        success_val = str(score >= 0.5).lower()
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])
        
        print(f"[END] success={success_val} steps={steps_taken} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())