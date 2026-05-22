#!/usr/bin/env python3
"""Comprehensive tool calling tests for MiMo2API — validates Cursor/CLI compatibility."""

import json
import sys
import httpx

BASE = "http://localhost:8080"
HEADERS = {
    "Authorization": "Bearer sk-default",
    "Content-Type": "application/json",
}
MODEL = "mimo-v2.5-pro"
TIMEOUT = 60.0

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command and return its output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute file path to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    }
]

passed = 0
failed = 0


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  PASS  {name}")
    except AssertionError as e:
        failed += 1
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        failed += 1
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


def chat(messages, tools=None, stream=False, tool_choice=None):
    body = {"model": MODEL, "messages": messages, "stream": stream}
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice
    r = httpx.post(f"{BASE}/v1/chat/completions", headers=HEADERS, json=body, timeout=TIMEOUT)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text}"
    return r.json()


def chat_stream(messages, tools=None, tool_choice=None):
    body = {"model": MODEL, "messages": messages, "stream": True}
    if tools:
        body["tools"] = tools
    if tool_choice:
        body["tool_choice"] = tool_choice

    chunks = []
    with httpx.stream("POST", f"{BASE}/v1/chat/completions", headers=HEADERS, json=body, timeout=TIMEOUT) as r:
        for line in r.iter_lines():
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
    return chunks


# ── Test 1: Basic tool calling (non-streaming) ──
def test_basic_tool_call():
    d = chat([{"role": "user", "content": "What files are in /tmp? Use bash."}], tools=TOOLS)
    c = d["choices"][0]
    assert c["finish_reason"] == "tool_calls", f"Expected tool_calls, got {c['finish_reason']}"
    tc = c["message"]["tool_calls"]
    assert tc and len(tc) >= 1, "Expected at least one tool call"
    assert tc[0]["type"] == "function"
    assert tc[0]["function"]["name"] == "bash"
    args = json.loads(tc[0]["function"]["arguments"])
    assert "command" in args, f"Expected 'command' in arguments, got {args}"
    assert c["message"]["content"] is None or c["message"]["content"] == ""


# ── Test 2: Streaming tool calling ──
def test_streaming_tool_call():
    chunks = chat_stream(
        [{"role": "user", "content": "List files in /etc using bash."}],
        tools=TOOLS
    )
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

    # First chunk should have role
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"

    # Find tool_calls chunk
    tool_call_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_call_chunks) >= 1, "No tool_calls chunks found in stream"

    tc = tool_call_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "bash"

    # Last chunk should have finish_reason=tool_calls
    finish_chunks = [c for c in chunks if c["choices"][0].get("finish_reason") == "tool_calls"]
    assert len(finish_chunks) >= 1, "No finish_reason=tool_calls chunk found"


# ── Test 3: Multi-turn tool conversation ──
def test_multi_turn():
    messages = [
        {"role": "user", "content": "Read /etc/hostname using bash"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_test_123",
                "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": '{"command": "cat /etc/hostname"}'
                }
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_test_123",
            "content": "mimo2api\n"
        }
    ]
    d = chat(messages, tools=TOOLS)
    c = d["choices"][0]
    assert c["finish_reason"] == "stop", f"Expected stop, got {c['finish_reason']}"
    assert c["message"]["content"], "Expected content in response"
    assert "mimo2api" in c["message"]["content"].lower(), f"Expected 'mimo2api' in response, got: {c['message']['content'][:100]}"


# ── Test 4: Multiple tools available — model picks correct one ──
def test_tool_selection():
    d = chat(
        [{"role": "user", "content": "Read the file /etc/hostname. Use the read_file tool."}],
        tools=TOOLS
    )
    c = d["choices"][0]
    assert c["finish_reason"] == "tool_calls"
    tc = c["message"]["tool_calls"]
    assert tc[0]["function"]["name"] == "read_file", f"Expected read_file, got {tc[0]['function']['name']}"
    args = json.loads(tc[0]["function"]["arguments"])
    assert "path" in args


# ── Test 5: No tools — normal response ──
def test_no_tools():
    d = chat([{"role": "user", "content": "Reply with just the word pong."}])
    c = d["choices"][0]
    assert c["finish_reason"] == "stop", f"Expected stop, got {c['finish_reason']}"
    assert c["message"]["content"], "Expected content"
    assert "pong" in c["message"]["content"].lower(), f"Expected 'pong' in response"
    assert c["message"].get("tool_calls") is None


# ── Test 6: tool_choice=required forces tool use ──
def test_tool_choice_required():
    d = chat(
        [{"role": "user", "content": "Say hello."}],
        tools=TOOLS,
        tool_choice="required"
    )
    c = d["choices"][0]
    assert c["finish_reason"] == "tool_calls", f"Expected tool_calls with required, got {c['finish_reason']}"
    assert c["message"]["tool_calls"]


# ── Test 7: tool_choice=none prevents tool use ──
def test_tool_choice_none():
    d = chat(
        [{"role": "user", "content": "What files are in /tmp? Use bash if you want."}],
        tools=TOOLS,
        tool_choice="none"
    )
    c = d["choices"][0]
    assert c["finish_reason"] == "stop", f"Expected stop with tool_choice=none, got {c['finish_reason']}"


# ── Test 8: tool_choice with specific function ──
def test_tool_choice_specific():
    d = chat(
        [{"role": "user", "content": "Do something with files."}],
        tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "read_file"}}
    )
    c = d["choices"][0]
    assert c["finish_reason"] == "tool_calls"
    tc = c["message"]["tool_calls"]
    assert tc[0]["function"]["name"] == "read_file", f"Expected read_file, got {tc[0]['function']['name']}"


# ── Test 9: Multiple tool calls in one response ──
def test_multiple_tool_calls():
    d = chat(
        [{"role": "user", "content": "Read both /etc/hostname AND /etc/os-release. Use read_file for each."}],
        tools=TOOLS
    )
    c = d["choices"][0]
    assert c["finish_reason"] == "tool_calls"
    assert c["message"]["tool_calls"]
    for tc in c["message"]["tool_calls"]:
        assert tc["type"] == "function"
        assert tc["function"]["name"] in ("read_file", "bash")


# ── Test 10: Streaming + multi-turn ──
def test_streaming_multi_turn():
    messages = [
        {"role": "user", "content": "Create a file /tmp/test.txt with content 'hello'. Use bash."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_stream_test",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "echo hello > /tmp/test.txt"}'}
            }]
        },
        {"role": "tool", "tool_call_id": "call_stream_test", "content": ""}
    ]
    chunks = chat_stream(messages, tools=TOOLS)

    # Should get a normal response (stop), not tool_calls
    finish = [c for c in chunks if c["choices"][0].get("finish_reason")]
    assert finish, "No finish chunk found"
    assert finish[0]["choices"][0]["finish_reason"] == "stop"


# ── Test 11: Valid JSON in tool call arguments ──
def test_valid_json_arguments():
    d = chat(
        [{"role": "user", "content": "Write 'test' to /tmp/test.json using write_file."}],
        tools=TOOLS
    )
    c = d["choices"][0]
    if c["finish_reason"] == "tool_calls":
        for tc in c["message"]["tool_calls"]:
            args = json.loads(tc["function"]["arguments"])  # Must parse without error
            assert isinstance(args, dict)


# ── Test 12: Response has proper OpenAI structure ──
def test_response_structure():
    d = chat([{"role": "user", "content": "Use bash to run echo hi"}], tools=TOOLS)
    assert "id" in d and d["id"].startswith("chatcmpl-")
    assert d["object"] == "chat.completion"
    assert "created" in d
    assert d["model"] == MODEL
    assert "choices" in d and len(d["choices"]) == 1
    assert "usage" in d
    assert d["usage"]["prompt_tokens"] > 0
    assert d["usage"]["completion_tokens"] > 0


# ── Test 13: Streaming response structure ──
def test_streaming_structure():
    chunks = chat_stream(
        [{"role": "user", "content": "Run echo hi with bash"}],
        tools=TOOLS
    )
    assert len(chunks) >= 2
    for c in chunks:
        assert c["object"] == "chat.completion.chunk"
        assert c["id"].startswith("chatcmpl-")
        assert len(c["choices"]) == 1


# ── Test 14: Auth still works ──
def test_auth_required():
    r = httpx.post(
        f"{BASE}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        timeout=10
    )
    assert r.status_code == 401


# ── Test 15: Bad API key ──
def test_bad_api_key():
    r = httpx.post(
        f"{BASE}/v1/chat/completions",
        headers={"Authorization": "Bearer wrong-key", "Content-Type": "application/json"},
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        timeout=10
    )
    assert r.status_code == 401


# ── Run all tests ──
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  MiMo2API Tool Calling Test Suite")
    print(f"{'='*60}\n")

    tests = [
        ("1.  Basic tool call (non-streaming)", test_basic_tool_call),
        ("2.  Streaming tool call", test_streaming_tool_call),
        ("3.  Multi-turn tool conversation", test_multi_turn),
        ("4.  Tool selection (picks correct tool)", test_tool_selection),
        ("5.  No tools — normal response", test_no_tools),
        ("6.  tool_choice=required forces tool use", test_tool_choice_required),
        ("7.  tool_choice=none prevents tool use", test_tool_choice_none),
        ("8.  tool_choice with specific function", test_tool_choice_specific),
        ("9.  Multiple tool calls in one response", test_multiple_tool_calls),
        ("10. Streaming + multi-turn", test_streaming_multi_turn),
        ("11. Valid JSON in tool call arguments", test_valid_json_arguments),
        ("12. Response has proper OpenAI structure", test_response_structure),
        ("13. Streaming response structure", test_streaming_structure),
        ("14. Auth required (no key = 401)", test_auth_required),
        ("15. Bad API key = 401", test_bad_api_key),
    ]

    for name, fn in tests:
        test(name, fn)

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'='*60}\n")

    if failed:
        print("Some tests failed. Not ready for Cursor/CLI use.")
        sys.exit(1)
    else:
        print("All tests passed! Ready for Cursor/CLI integration.")
        sys.exit(0)
