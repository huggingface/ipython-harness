from ipyh import Harness
from openai import OpenAI

url = 'http://localhost:8080/v1'
model_name = 'unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL'

model = OpenAI(
    base_url=url,
    api_key='EMPTY'
)

harness = Harness()
agent = harness + model

task = 'Please list the files that are available in the current directory'

agent.run_task(task, debug=True)

_ = agent.run_step()

