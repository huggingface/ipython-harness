# ipython-harness

A minimal and experimental agentic harness, based purely on `ipython` used for evaluation and benchmarking purpose (see [this post]() for more details) 

## Installation

```
uv pip install git+https://github.com/huggingface/ipython-harness
```

## Usage

An example usage is provided in `example.py` - you can use any openai compatible server - for example you can run a in a different shell:

```bash
llama-server -hf unsloth/Qwen3.5-9B-GGUF:UD-Q4_K_XL --port 8080 --host 0.0.0.0 -ngl 99
```

and in `example.py`:

```py
from openai import OpenAI
from ipyh import Harness

url = "http://localhost:8080/v1"

# `agent` picks up the first model_name available on the server
model = OpenAI(base_url=url, api_key="EMPTY")

task = "Please analyse the current codebase and write me a minimal README.md file that guides the user on the purpose of this library and how to use it with `example.py`"

agent = Harness() + model

message = agent.run_task_sync(task, approve_commands=True)
print(message.content)
```

`approve_commands` will ask you to prompt `y` or `n` upon inspecting the proposed tool call of the model.

## Evaluation

There is a `harbor` integration inside `integrations/`. We will provide later reproducible scripts to obtain our evaluations.

## Contributing

Feel free to contribute to make the harness better or more efficient.