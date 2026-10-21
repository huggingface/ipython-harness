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

### Kernel tools

Besides `ipython`, a harness can expose Python functions as their own tools. Each one is defined in the kernel when it starts, so it runs with the same state and files as the model's cells:

```py
harness = Harness(tools=[{
    "name": "word_count",
    "description": "Count the words in a file.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    "code": "def word_count(path):\n    return len(open(path).read().split())",
}])
```

The same settings can live in a JSON file with optional `system_prompt` and `tools` keys: `Harness.from_config("harness.json")`, or `python -m ipyh --harness-config harness.json ...`. Keeping that file under version control tracks how a harness changes over time.

## Evaluation

There is a `harbor` integration inside `integrations/`. We will provide later reproducible scripts to obtain our evaluations.

The [Dino reverse-engineering experiment](experiments/dino/README.md) compares one large agent with a large reviewer plus smaller coding agents. Its oracle, visual adapter, Hub workflow, and token accounting live outside the core package. See the [HF Jobs runbook](experiments/dino/RUN.md) to launch it.

The [Neon Heist experiment](experiments/heist/RUN.md) uses the same solo/team comparison on a playable turn based Python and Gradio game. Its [harness history report](experiments/heist/HARNESS_HISTORY.md) indexes lead-created harness revisions from saved runs; regenerate it with `python -m experiments.heist.harness_history --root . --output experiments/heist/HARNESS_HISTORY.md`.

## Contributing

Feel free to contribute to make the harness better or more efficient.
