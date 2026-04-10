import asyncio
import os
import re
from openai import OpenAI
from client import OrangePigeonEnv
from models import OrangePigeonAction

async def main() -> None:
    # EXACTLY AS REQUESTED BY SCALER GRADER (No fallbacks, single line)
    client = OpenAI(
        base_url=os.environ["API_BASE_URL"],
        api_key=os.environ["API_KEY"]
    )

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    openenv_url = os.environ.get("OPENENV_URL")
    image_name = os.environ.get("LOCAL_IMAGE_NAME")

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

        for step in range(1, 11):
            if result.observation.done: break

            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"State: {last_state}. Output 1 integer (0,1,2)."}],
                max_tokens=10
            )
            
            action_text = completion.choices[0].message.content.strip()
            match = re.search(r'\d+', action_text)
            action_int = int(match.group(0)) if match else 0

            result = await env.step(OrangePigeonAction(action=action_int))
            rewards.append(result.observation.reward or 0.0)
            last_state = result.observation.state
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if env: await env.close()

if __name__ == "__main__":
    asyncio.run(main())