"""Utility functions"""

import re
import json
import uuid
from typing import Optional, List, Tuple
from .config import MimoAccount


def parse_curl(curl_command: str) -> Optional[MimoAccount]:
    """
    Parse cURL command to extract MiMo account credentials.

    Args:
        curl_command: cURL command string

    Returns:
        MimoAccount object or None
    """
    account = {
        'service_token': '',
        'user_id': '',
        'xiaomichatbot_ph': ''
    }

    # Extract cookies (supports multiple formats)
    cookie_match = re.search(r"(?:-b|--cookie)\s+'([^']+)'", curl_command)
    if not cookie_match:
        cookie_match = re.search(r'(?:-b|--cookie)\s+"([^"]+)"', curl_command)
    if not cookie_match:
        cookie_match = re.search(r"-H\s+'[Cc]ookie:\s*([^']+)'", curl_command)
    if not cookie_match:
        cookie_match = re.search(r'-H\s+"[Cc]ookie:\s*([^"]+)"', curl_command)
    if not cookie_match:
        return None

    cookies = cookie_match.group(1)

    # Extract serviceToken
    service_token_match = re.search(r'serviceToken="([^"]+)"', cookies)
    if service_token_match:
        account['service_token'] = service_token_match.group(1)

    # Extract userId
    user_id_match = re.search(r'userId=(\d+)', cookies)
    if user_id_match:
        account['user_id'] = user_id_match.group(1)

    # Extract xiaomichatbot_ph
    ph_match = re.search(r'xiaomichatbot_ph="([^"]+)"', cookies)
    if ph_match:
        account['xiaomichatbot_ph'] = ph_match.group(1)

    # Validate required fields
    if not account['service_token']:
        return None

    return MimoAccount(**account)


def safe_utf8_len(text: str, max_len: int) -> int:
    """
    Safe UTF-8 string length calculation to avoid truncating multi-byte characters.

    Args:
        text: Input string
        max_len: Maximum length

    Returns:
        Safe truncation length
    """
    if max_len <= 0 or max_len >= len(text):
        return len(text)

    # Python 3 strings are Unicode, no special UTF-8 boundary handling needed
    # Kept for consistency with the Go version
    return max_len


def build_query_from_messages(messages: list, max_messages: int = 10, max_content_len: int = 4000) -> str:
    """
    Build query string from message list.

    Args:
        messages: List of messages
        max_messages: Maximum number of messages
        max_content_len: Maximum length per message

    Returns:
        Query string
    """
    # Keep only the last N messages
    if len(messages) > max_messages:
        messages = messages[-max_messages:]

    query_parts = []
    for msg in messages:
        content = msg.content
        # Truncate overly long content
        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."
        query_parts.append(f"{msg.role}: {content}")

    return "\n".join(query_parts)

# ---- Tool calling support ----

_TOOL_CALL_OPEN = "<" + "tool_call>"
_TOOL_CALL_CLOSE = "<" + "/tool_call>"

import re as _re
_PARSE_PATTERN = _re.compile(
    r"<" + "tool_call>(.*?)<" + "/tool_call>",
    _re.DOTALL,
)


def build_tool_system_prompt(tools, tool_choice=None):
    """Serialize OpenAI tool definitions into a system-prompt fragment."""
    tools_text = ""
    for i, tool in enumerate(tools):
        if isinstance(tool, dict):
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
        else:
            func = tool.function if hasattr(tool, "function") else {}
            name = func.name if hasattr(func, "name") else "unknown"
            desc = func.description if hasattr(func, "description") else ""
            params = func.parameters if hasattr(func, "parameters") else {}
        tools_text += f"\nTool {i + 1}: {name}\nDescription: {desc}\n"
        tools_text += f"Parameters: {json.dumps(params, indent=2)}\n"

    prompt = (
        "You have access to the following tools. To call a tool, respond with\n"
        "a " + _TOOL_CALL_OPEN + " tag containing a JSON object with 'name' and 'arguments'.\n"
        "You may call multiple tools by using multiple " + _TOOL_CALL_OPEN + "..." + _TOOL_CALL_CLOSE + " blocks.\n"
        "\nAvailable tools:\n" + tools_text + "\n"
        "Rules:\n"
        "- Use valid JSON inside the " + _TOOL_CALL_OPEN + " tag.\n"
        "- The 'arguments' field must be a JSON object matching the tool's parameter schema.\n"
        "- After all tool calls, do not add any other text.\n"
        "- If you do not need to call any tool, respond normally without " + _TOOL_CALL_OPEN + " tags.\n"
    )

    if tool_choice == "none":
        prompt += "\nThe user has disabled tool calling. Respond without using any tools.\n"
    elif tool_choice == "required":
        prompt += "\nYou MUST call at least one tool. Do not respond with plain text.\n"
    elif isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
        prompt += f"\nYou MUST call the tool named '{tool_choice}'.\n"

    return prompt


def parse_tool_calls_from_response(text):
    """Extract tool calls from the model's text response.

    Returns (remaining_content, tool_calls_list).  If no tool calls are found
    the original text is returned unchanged and tool_calls_list is None.
    """
    matches = _PARSE_PATTERN.findall(text)
    if not matches:
        return text, None

    tool_calls = []
    for raw in matches:
        raw = raw.strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = data.get("name", "")
        arguments = data.get("arguments", {})
        if not name:
            continue
        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments) if isinstance(arguments, dict) else str(arguments),
            },
        })

    if not tool_calls:
        return text, None

    cleaned = _PARSE_PATTERN.sub("", text).strip()
    return (cleaned if cleaned else ""), tool_calls


def build_query_with_tools(messages, tools, tool_choice=None, max_messages=10, max_content_len=4000):
    """Build query string from messages, injecting tool instructions."""
    tool_prompt = build_tool_system_prompt(tools, tool_choice)
    query_parts = [f"system: {tool_prompt}"]

    for msg in messages:
        content = msg.content or ""

        if msg.role == "assistant" and getattr(msg, "tool_calls", None):
            call_lines = []
            for tc in msg.tool_calls:
                if isinstance(tc, dict):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", "")
                else:
                    func = tc.function if hasattr(tc, "function") else tc.get("function", {})
                    if isinstance(func, dict):
                        name = func.get("name", "")
                        args = func.get("arguments", "")
                    else:
                        name = getattr(func, "name", "")
                        args = getattr(func, "arguments", "")
                call_lines.append(f"{_TOOL_CALL_OPEN}{{\"name\": \"{name}\", \"arguments\": {args}}}{_TOOL_CALL_CLOSE}")
            content = "\n".join(call_lines) if call_lines else content

        if msg.role == "tool":
            tid = getattr(msg, "tool_call_id", "") or ""
            content = f"[Tool result for {tid}]: {content}"

        if len(content) > max_content_len:
            content = content[:max_content_len] + "..."

        if content:
            query_parts.append(f"{msg.role}: {content}")

    if len(query_parts) > max_messages + 1:
        query_parts = [query_parts[0]] + query_parts[-(max_messages):]

    return "\n".join(query_parts)

