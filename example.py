"""Run one task with a local kernel. The model's code runs on this machine: use a throwaway container.

Synchronous here; from async code, use `openai.AsyncOpenAI` and `await agent.run_task(task)`.
"""

from openai import OpenAI
from ipyh import Harness

url = "http://localhost:8080/v1"
model = OpenAI(base_url=url, api_key="EMPTY")

task = "Please analyse the current codebase and write me a minimal README.md file that guides the user on the purpose of this library and how to use it with `example.py`"

agent = Harness() + model

message = agent.run_task_sync(task, approve_commands=True)
print(message.content)
