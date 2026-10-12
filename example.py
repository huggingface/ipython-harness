"""Run one task with a local kernel. The model's code runs on this machine: use a throwaway container.

Synchronous here; from async code, use `openai.AsyncOpenAI` and `await agent.run_task(task)`.
"""

from openai import OpenAI

from ipyh import Harness

url = "http://localhost:8080/v1"
model_name = "unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL"

task = "Please list all the single files that are available in the current directory and write the summary in SUMMARY.md"

agent = Harness() + OpenAI(base_url=url, api_key="EMPTY")
agent.model_name = model_name
message = agent.run_task_sync(task)
print(message.content)
