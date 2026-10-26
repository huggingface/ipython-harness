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

## Skills

The harness supports [Agent Skills](https://agentskills.io): folders with a `SKILL.md` whose frontmatter has a `name` and a `description`. Pass SKILL.md files, skill folders, or folders of skill folders (e.g. `~/.claude/skills`):

```py
agent = Harness(skills=["~/.claude/skills", "./my-skill"]) + model
```

The CLI takes `--skills PATH [PATH ...]`, and a `--harness-config` JSON file can include a `"skills"` list. As in pi or deepagents, only each skill's name, description and path go in the system prompt (in an `<available_skills>` block). The model reads a full `SKILL.md` with the kernel's `view()` helper when a task matches it, so there is no extra tool. The kernel must be able to read the skill folders, which is always true for the local kernel.

## Harness folders

A harness can be saved and loaded as a folder:

```
my-harness/
  harness.json        {"system_prompt_file": "system_prompt.md", "tools": [...], "skills": ["skills"],
                       "cell_timeout_sec": 300, "agent": {"max_iteration": 30}}
  system_prompt.md    must contain `{python}`
  tools/<name>.py     kernel tools: functions defined in the kernel and offered as their own tools
  skills/<name>/SKILL.md
```

```py
harness = Harness.from_config("my-harness")   # agent settings apply to `harness + model`
harness.save("copy-of-my-harness")
```

`agent.stop_tools` names kernel tools whose successful call ends the task (e.g. a `submit_answer` tool), so the model is not asked for a closing reply. A call that raises does not end it. The CLI takes the folder as `--harness-config my-harness`. A single JSON file with inline `system_prompt` and tool `code` still works.

## Optimizing a harness for a task suite

[ipyh-optimization](https://github.com/huggingface/ipyh-optimization) builds an ipyh harness folder for a suite of tasks, adapting [Harness²](https://github.com/google-research/harness2), without training. It also compares harnesses on held-out tasks.

## Evaluation

There is a `harbor` integration inside `integrations/`. We will provide later reproducible scripts to obtain our evaluations.

## Contributing

Feel free to contribute to make the harness better or more efficient.