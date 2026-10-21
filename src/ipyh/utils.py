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
from typing import Optional

import json
import re
import secrets

from ipyh.normalizers import truncate


_XML_CALL = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S)
_XML_PARAMETER = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)
_JSON_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def recover_tool_calls(text: str, tool_list: list) -> list:
    """
    Complete tool calls left as text that the server did not parse (some providers leave them in
    the reasoning), in the XML `<function=...><parameter=...>` or the JSON `<tool_call>` format.
    """
    schemas = {
        tool["function"]["name"]: tool["function"].get("parameters", {}).get("properties", {})
        for tool in tool_list
    }

    def typed(name, key, value):
        # XML values are text: decode them only where the tool's schema expects something else
        if schemas.get(name, {}).get(key, {}).get("type", "string") == "string":
            return value
        try:
            return json.loads(value)
        except ValueError:
            return value

    calls = [
        (name, {key: typed(name, key, value) for key, value in _XML_PARAMETER.findall(body)})
        for name, body in _XML_CALL.findall(text)
    ]
    for raw in [] if calls else _JSON_CALL.findall(text):
        try:
            call = json.loads(raw)
            calls.append((call["name"], call.get("arguments") or {}))
        except (ValueError, KeyError, TypeError):
            pass
    return [
        {
            "id": f"call_{secrets.token_hex(6)}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }
        for name, arguments in calls
        if name in schemas
    ]


def render_transcript(messages: list, limit: Optional[int] = None) -> str:
    """Messages as plain text, each cell and output cut to `limit` characters (None: whole)."""
    parts = []
    for message in messages:
        text = message.get("content") or ""
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            try:
                arguments = json.loads(arguments).get("code", arguments)
            except (ValueError, AttributeError):
                pass
            text += f"\n[{call['function']['name']}]\n{arguments}"
        parts.append(f"[{message['role']}]\n{truncate(text, limit) if limit else text}")
    return "\n\n".join(parts)
