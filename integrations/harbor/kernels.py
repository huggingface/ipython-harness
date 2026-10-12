# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Running the model's cells inside a sandbox.

RemoteKernel runs an IPython kernel inside a sandbox through async `exec` functions, so it works
with any sandbox that can run a shell command (Harbor, `docker exec`, ssh, ...). It has `python`
(the interpreter path named in the system prompt, or None for a private one),
`async run(code, timeout) -> str` and `async close()`. Without one, the agent runs cells in a
kernel on this machine (see Agent.run_task).
"""

from __future__ import annotations

import base64
import json
import shlex
import uuid
from typing import Awaitable, Callable, Optional, Tuple

from ipyh.constants import CELL_TIMEOUT_SEC
from ipyh.normalizers import QUIET_ENV

# exec(command, timeout_sec) -> (stdout, stderr, return_code)
Exec = Callable[[str, Optional[int]], Awaitable[Tuple[Optional[str], Optional[str], int]]]


ROOT = "/opt/ipy"  # kernel venv (and uv), kept apart from the task's own environment
RUN = "/tmp/ipy"  # kernel connection file, pid, logs and cell scratch files

# Installs the kernel. Prints the task interpreter it attached to, or nothing on fallback.
INSTALL = f"""set -euo pipefail
mkdir -p {ROOT}/bin {RUN} && chmod 777 {RUN}
curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL={ROOT}/bin sh >/dev/null
export UV_PYTHON_INSTALL_DIR={ROOT}/python
uv={ROOT}/bin/uv venv={ROOT}/venv
sys=$(command -v python3 || true)
if [ -n "$sys" ] && "$sys" -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null \\
   && $uv venv --quiet --python "$sys" $venv \\
   && $uv pip install --quiet --python $venv/bin/python ipykernel jupyter_client; then
  "$sys" -c 'import site; print("\\n".join(site.getsitepackages() + [site.getusersitepackages()]))' \\
    > {ROOT}/task-site-packages.txt
  echo "$sys"
else
  rm -rf $venv
  $uv venv --quiet --seed --python 3.12 $venv >&2
  $uv pip install --quiet --python $venv/bin/python ipykernel jupyter_client >&2
fi
chmod -R a+rX {ROOT}
"""

# Runs in the kernel once it is up. Puts the task interpreter's site-packages ahead of the kernel
# venv's, so user code imports the task's versions of packages ipykernel also depends on (modules
# the kernel already loaded stay as they are), and reports the task interpreter as the kernel's
# own, so `%pip`, `sys.executable` and subprocesses target the task's Python.
STARTUP = f"""def _use_task_python():
    import site, sys
    kernel = list(sys.path)
    for path in open("{ROOT}/task-site-packages.txt").read().splitlines():
        site.addsitedir(path)
    at = next((i for i, p in enumerate(kernel) if p.startswith("{ROOT}/")), len(kernel))
    sys.path[:] = kernel[:at] + [p for p in sys.path if p not in kernel] + kernel[at:]
    sys.executable = {{path!r}}
_use_task_python()
del _use_task_python
"""

# Runs inside the sandbox with the kernel venv's Python: executes one cell, prints its output.
RUN_CELL = r'''
import os, re, signal, sys
from jupyter_client import BlockingKernelClient

conn, code_path, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
kc = BlockingKernelClient(connection_file=conn)
kc.load_connection_file()
kc.start_channels()
kc.wait_for_ready(timeout=120)
out = []

def hook(msg):
    kind, content = msg["msg_type"], msg["content"]
    if kind == "stream":
        out.append(content["text"])
    elif kind in ("execute_result", "display_data"):
        out.append(content["data"].get("text/plain", "") + "\n")
    elif kind == "error":
        out.append("\n".join(content["traceback"]) + "\n")

try:
    kc.execute_interactive(open(code_path).read(), output_hook=hook, timeout=timeout, allow_stdin=False)
except TimeoutError:
    os.kill(int(open(sys.argv[4]).read()), signal.SIGINT)
    out.append(f"\n[cell timed out after {timeout:.0f}s; kernel interrupted, state kept]\n")
text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", "".join(out))
sys.stdout.write(text or "[no output]")
'''

# Starts the kernel in the background from the caller's working directory and waits for it.
LAUNCH = (
    "$(command -v setsid || true) "  # detach where available; minimal images may lack it
    f"nohup {ROOT}/venv/bin/python -m ipykernel_launcher -f {RUN}/kernel.json "
    f"--IPKernelApp.exec_files={RUN}/startup.py "
    f"> {RUN}/kernel.log 2>&1 < /dev/null & echo $! > {RUN}/kernel.pid; "
    f"for i in $(seq 120); do [ -s {RUN}/kernel.json ] && exit 0; sleep 1; done; "
    f"cat {RUN}/kernel.log; exit 1"
)


def _write_file(path: str, content: str) -> str:
    encoded = base64.b64encode(content.encode()).decode()
    return f"echo {shlex.quote(encoded)} | base64 -d > {path}"


class RemoteKernel:
    """An IPython kernel inside a sandbox, driven through shell commands.

    `exec` runs each cell. `install()` also takes the functions to run the install as root and
    to launch the kernel as the task's user, from the task's working directory.
    """

    def __init__(self, exec: Exec, cell_timeout_sec: int = CELL_TIMEOUT_SEC, quiet: bool = True) -> None:
        self.exec = exec
        self.cell_timeout_sec = cell_timeout_sec
        self.quiet = quiet
        self.python: Optional[str] = None

    async def install(self, root_exec: Optional[Exec] = None, agent_exec: Optional[Exec] = None) -> "RemoteKernel":
        root_exec, agent_exec = root_exec or self.exec, agent_exec or self.exec
        stdout, stderr, rc = await root_exec(INSTALL, None)
        if rc != 0:
            raise RuntimeError(f"kernel install failed ({rc}): {stderr or stdout}")
        self.python = (stdout or "").strip() or None
        startup = STARTUP.format(path=self.python) if self.python else ""
        for name, content in (("run_cell.py", RUN_CELL), ("startup.py", startup)):
            stdout, stderr, rc = await root_exec(_write_file(f"{RUN}/{name}", content), None)
            if rc != 0:
                raise RuntimeError(f"writing {RUN}/{name} failed ({rc}): {stderr or stdout}")
        stdout, stderr, rc = await agent_exec(LAUNCH, None)
        if rc != 0:
            raise RuntimeError(f"kernel launch failed ({rc}): {stderr or stdout}")
        if self.quiet:
            await self.run(f"import os; os.environ.update({json.dumps(QUIET_ENV)})")
        return self

    async def run(self, code: str, timeout: Optional[float] = None) -> str:
        timeout = int(timeout or self.cell_timeout_sec)
        cell = f"{RUN}/cell-{uuid.uuid4().hex}.py"
        stdout, stderr, rc = await self.exec(
            f"{_write_file(cell, code)} && "
            f"{ROOT}/venv/bin/python {RUN}/run_cell.py {RUN}/kernel.json {cell} {timeout} {RUN}/kernel.pid; "
            f"rm -f {cell}",
            timeout + 120,
        )
        text = (stdout or "") + (f"\n[stderr]\n{stderr}" if stderr else "")
        if rc != 0:
            text += f"\n[cell runner exited with {rc}]"
        return text

    async def close(self) -> None:
        """The kernel lives as long as the sandbox."""
