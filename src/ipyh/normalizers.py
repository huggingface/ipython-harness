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

"""Remove run-to-run noise from tool output before a model sees it.

Two layers, both minimal and conservative:

QUIET_ENV prevents noise at the source. Exported into the agent's shell, it turns off progress
bars and colour in common tools, so most of the noise is never printed.

normalize() cleans what still gets through, in one pass of precompiled regexes:
  1. terminal control: ANSI/ECMA-48 escape sequences (CSI incl. private modes such as the
     `\\x1b[?25l` cursor toggles that pip/rich print, OSC, 2-byte ESC), then carriage-return
     overwrites resolved as a terminal would (keep what follows the last `\\r` on each line), so a
     redrawn progress bar collapses to its final frame;
  2. progress-bar lines (runs of bar glyphs), which carry nothing beyond the line before them;
  3. values that differ on every run and are never task data: transfer rates, Python object
     addresses, and the random suffixes of temporary names.

Timestamps, dates and durations are left alone on purpose: tasks can depend on them (log
analysis, file ages, timing limits), and masking them would hide information.
"""

from __future__ import annotations

import re

from ipyh.constants import MAX_OUTPUT_CHARS

QUIET_ENV = {
    "PIP_PROGRESS_BAR": "off",  # pip's rich progress bar (the speed/eta line)
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",  # "a new release of pip is available" notice
    "TQDM_DISABLE": "1",  # tqdm >= 4.66 reads TQDM_* defaults from the environment
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "UV_NO_PROGRESS": "1",
    "npm_config_progress": "false",
    "NO_COLOR": "1",  # https://no-color.org
}

_ANSI = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI, including private modes (\x1b[?25l)
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC (titles, hyperlinks)
    r"|\x1b[@-Z\\-_]"  # other 2-byte escapes
)
# Text before the last \r that is followed by more text on the same line (a terminal overwrites it);
# a lone trailing \r overwrites nothing.
_OVERWRITTEN = re.compile(r"^[^\n]*\r(?=[^\n])", re.M)
_LONE_CR = re.compile(r"\r(?=\n|\Z)")
# Lines drawn with progress-bar glyphs. `#`, `=`, `-`, `_` are excluded: pytest separators and
# markdown rules use them and carry information.
_BAR_LINE = re.compile(r"^[^\n]*[━█▉▊▋▌▍▎▏╸╺■]{10,}[^\n]*\n?", re.M)
_MASKS = [
    (re.compile(r"\b\d+(?:\.\d+)?\s?[kKMGT]i?B/s\b"), "<rate>"),
    (re.compile(r"(\bat 0x)[0-9a-fA-F]{6,}\b"), r"\1<addr>"),  # <Foo object at 0x7f3a…>
    (re.compile(r"((?:/tmp|/var/tmp)/[\w.-]*?(?:tmp|-))[a-z0-9_]{8}\b"), r"\1<tmp>"),
]


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS, saved_to: str = None) -> str:
    """Keep the head and tail of output longer than `limit` characters, naming where the full text is."""
    if len(text) <= limit:
        return text
    half = limit // 2
    where = f"; full output in {saved_to}" if saved_to else ""
    return f"{text[:half]}\n[… {len(text) - limit} chars truncated{where} …]\n{text[-half:]}"


def python_default_normalizer(text: str) -> str:
    """Deterministic, idempotent, and only ever removes or masks text."""
    text = _ANSI.sub("", text).replace("\r\n", "\n")
    text = _LONE_CR.sub("", _OVERWRITTEN.sub("", text))
    text = _BAR_LINE.sub("", text)
    for pattern, replacement in _MASKS:
        text = pattern.sub(replacement, text)
    return text

