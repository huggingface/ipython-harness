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
"""Agent Skills (https://agentskills.io): folders with a `SKILL.md` whose frontmatter has a `name`
and a `description`. Only those two and the file's path go in the system prompt; the model reads
the body with `view()` when a task matches (progressive disclosure, as in pi or deepagents).
"""

import warnings
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Union

from ipyh.constants import SKILLS_PROMPT


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: str  # absolute path of the SKILL.md


def load_skills(paths: Iterable[Union[str, Path]]) -> List[Skill]:
    """The skills under `paths`. Each path is a SKILL.md, a skill folder, or a folder of skill
    folders (e.g. `~/.claude/skills`). When two skills share a name, the first one is kept."""
    found: Dict[str, Skill] = {}
    for path in paths:
        path = Path(path).expanduser().resolve()
        if path.is_file():
            files = [path]
        elif (path / "SKILL.md").is_file():
            files = [path / "SKILL.md"]
        elif path.is_dir():
            files = sorted(path.glob("*/SKILL.md"))
        else:
            raise FileNotFoundError(f"no skill at {path}")
        for file in files:
            skill = parse_skill(file)
            if skill is None:
                continue
            if skill.name in found:
                warnings.warn(f"skill {skill.name!r} at {file} skipped: already loaded from {found[skill.name].path}")
                continue
            found[skill.name] = skill
    return list(found.values())


def parse_skill(file: Path):
    """The skill a SKILL.md describes, or None (with a warning) when it has no description."""
    meta = _frontmatter(file.read_text(errors="replace"))
    description = " ".join(meta.get("description", "").split())
    if not description:
        warnings.warn(f"{file} skipped: its frontmatter has no `description`")
        return None
    # The spec asks for `name` to match the folder; the folder is the fallback
    return Skill(name=meta.get("name") or file.parent.name, description=description, path=str(file))


def render_skills(skills: List[Skill]) -> str:
    """The system prompt section listing `skills` ("" when there are none)."""
    if not skills:
        return ""
    entries = "\n".join(
        "<skill>\n"
        + "".join(f"<{tag}>{escape(value, quote=False)}</{tag}>\n" for tag, value in (
            ("name", s.name), ("description", s.description), ("location", s.path)
        ))
        + "</skill>"
        for s in skills
    )
    return f"{SKILLS_PROMPT}\n\n<available_skills>\n{entries}\n</available_skills>"


def _frontmatter(text: str) -> Dict[str, str]:
    """The top-level `key: value` pairs of a `---` YAML frontmatter block. Handles plain and quoted
    scalars and `>` / `|` block scalars, which is what skill frontmatter uses; nested keys are skipped."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta, key = {}, None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        if line[:1] in (" ", "\t") or not line.strip():
            # Continuation of a block scalar or a multi-line plain scalar
            if key is not None:
                meta[key] = f"{meta[key]} {line.strip()}".strip()
            continue
        name, sep, value = line.partition(":")
        if not sep:
            key = None
            continue
        key, value = name.strip(), value.strip()
        if value in (">", "|", ">-", "|-", ">+", "|+"):
            value = ""
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        meta[key] = value
    return meta
