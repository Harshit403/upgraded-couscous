"""API routes"""

import time
import uuid
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse
from .models import (
    OpenAIRequest, OpenAIResponse, OpenAIChoice, OpenAIMessage,
    OpenAIDelta, OpenAIUsage, ParseCurlRequest, TestAccountRequest,
    ToolCall, FunctionCall
)
from .config import config_manager, MimoAccount
from .mimo_client import MimoClient
from .utils import parse_curl, build_query_from_messages, build_query_with_tools, parse_tool_calls_from_response

router = APIRouter()


def validate_api_key(authorization: Optional[str]) -> bool:
    """Validate API key"""
    if not authorization:
        return False

    # Strip "Bearer " prefix
    key = authorization.replace("Bearer ", "").strip()
    return config_manager.validate_api_key(key)


@router.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIRequest,
    authorization: Optional[str] = Header(None)
):
    """OpenAI-compatible chat endpoint"""

    # Validate API key
    if not validate_api_key(authorization):
        raise HTTPException(status_code=401, detail={"error": {"message": "invalid api key"}})

    # Get next MiMo account
    account = config_manager.get_next_account()
    if not account:
        raise HTTPException(status_code=503, detail={"error": {"message": "no mimo account"}})

    # Build query string — inject tool instructions if tools are present
    has_tools = bool(request.tools)
    if has_tools:
        tool_choice = request.tool_choice
        if isinstance(tool_choice, dict):
            tool_choice = tool_choice.get("function", {}).get("name")
        elif tool_choice not in ("auto", "none", "required", None):
            tool_choice = str(tool_choice)
        query = build_query_with_tools(request.messages, request.tools, tool_choice)
    else:
        query = build_query_from_messages(request.messages)

    # Check if deep thinking is enabled
    thinking = bool(request.reasoning_effort)

    # Create MiMo client
    client = MimoClient(account)

    # Streaming response
    if request.stream:
        return StreamingResponse(
            stream_response(client, query, thinking, request.model, has_tools),
            media_type="text/event-stream"
        )

    # Non-streaming response
    try:
        content, think_content, usage = await client.call_api(query, thinking)

        # Parse tool calls from the response
        remaining_content, tool_calls = parse_tool_calls_from_response(content)

        if tool_calls:
            # Return tool_calls response
            response = OpenAIResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[
                    OpenAIChoice(
                        index=0,
                        message=OpenAIMessage(
                            role="assistant",
                            content=remaining_content or None,
                            tool_calls=[ToolCall(**tc) for tc in tool_calls]
                        ),
                        finish_reason="tool_calls"
                    )
                ],
                usage=OpenAIUsage(
                    prompt_tokens=usage.get("promptTokens", 0),
                    completion_tokens=usage.get("completionTokens", 0),
                    total_tokens=usage.get("promptTokens", 0) + usage.get("completionTokens", 0)
                )
            )
        else:
            # Normal response
            full_content = content
            if think_content:
                full_content = f"<think>{think_content}</think>\n{content}"

            response = OpenAIResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[
                    OpenAIChoice(
                        index=0,
                        message=OpenAIMessage(role="assistant", content=full_content),
                        finish_reason="stop"
                    )
                ],
                usage=OpenAIUsage(
                    prompt_tokens=usage.get("promptTokens", 0),
                    completion_tokens=usage.get("completionTokens", 0),
                    total_tokens=usage.get("promptTokens", 0) + usage.get("completionTokens", 0)
                )
            )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": {"message": str(e)}})


async def stream_response(client: MimoClient, query: str, thinking: bool, model: str, has_tools: bool = False):
    """Streaming response generator"""

    msg_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    # Send initial role delta
    yield f"data: {json.dumps(OpenAIResponse(id=msg_id, object='chat.completion.chunk', created=int(time.time()), model=model, choices=[OpenAIChoice(index=0, delta=OpenAIDelta(role='assistant'))]).dict())}\n\n"

    # When tools are present, buffer the full response to parse tool calls
    if has_tools:
        full_text = ""
        try:
            async for sse_data in client.stream_api(query, thinking):
                content = sse_data.get("content", "")
                if content:
                    full_text += content

            # Parse tool calls from the full response
            full_text = full_text.replace("\x00", "")
            remaining_content, tool_calls = parse_tool_calls_from_response(full_text)

            if tool_calls:
                # Emit tool_calls as delta chunks
                for i, tc in enumerate(tool_calls):
                    tc_obj = ToolCall(
                        id=tc["id"],
                        type="function",
                        function=FunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"]
                        )
                    )
                    delta = OpenAIDelta(tool_calls=[tc_obj])
                    chunk = OpenAIResponse(
                        id=msg_id,
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=model,
                        choices=[OpenAIChoice(index=0, delta=delta)]
                    )
                    yield f"data: {json.dumps(chunk.dict())}\n\n"

                # Send end with tool_calls finish_reason
                final_chunk = OpenAIResponse(
                    id=msg_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[OpenAIChoice(index=0, delta=OpenAIDelta(), finish_reason="tool_calls")]
                )
                yield f"data: {json.dumps(final_chunk.dict())}\n\n"
            else:
                # No tool calls — emit as normal content
                if remaining_content:
                    chunk = OpenAIResponse(
                        id=msg_id,
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=model,
                        choices=[OpenAIChoice(index=0, delta=OpenAIDelta(content=remaining_content))]
                    )
                    yield f"data: {json.dumps(chunk.dict())}\n\n"

                final_chunk = OpenAIResponse(
                    id=msg_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[OpenAIChoice(index=0, delta=OpenAIDelta(), finish_reason="stop")]
                )
                yield f"data: {json.dumps(final_chunk.dict())}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            error_chunk = {"error": {"message": str(e)}}
            yield f"data: {json.dumps(error_chunk)}\n\n"

        return

    # Normal streaming (no tools) — stream with think tag parsing
    buffer = ""
    in_think = False

    try:
        async for sse_data in client.stream_api(query, thinking):
            content = sse_data.get("content", "")
            if not content:
                continue

            buffer += content
            text = buffer.replace("\x00", "")

            # Process<think>tags
            while True:
                if not in_think:
                    # Find>tag
                    idx = text.find("<think>")
                    if idx != -1:
                        # Send content before<think>
                        if idx > 0:
                            chunk = OpenAIResponse(
                                id=msg_id,
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                model=model,
                                choices=[OpenAIChoice(index=0, delta=OpenAIDelta(content=text[:idx]))]
                            )
                            yield f"data: {json.dumps(chunk.dict())}\n\n"

                        in_think = True
                        text = text[idx + 7:]
                        continue

                    # Keep last 7 chars in case<think>is truncated
                    safe = len(text) - 7
                    if safe > 0:
                        chunk = OpenAIResponse(
                            id=msg_id,
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=model,
                            choices=[OpenAIChoice(index=0, delta=OpenAIDelta(content=text[:safe]))]
                        )
                        yield f"data: {json.dumps(chunk.dict())}\n\n"
                        text = text[safe:]
                    break

                else:
                    # Find>tag
                    idx = text.find("</think>")
                    if idx != -1:
                        # Send reasoning content before</think>
                        if idx > 0:
                            chunk = OpenAIResponse(
                                id=msg_id,
                                object="chat.completion.chunk",
                                created=int(time.time()),
                                model=model,
                                choices=[OpenAIChoice(index=0, delta=OpenAIDelta(reasoning=text[:idx]))]
                            )
                            yield f"data: {json.dumps(chunk.dict())}\n\n"

                        in_think = False
                        text = text[idx + 8:]
                        continue

                    # Keep last 8 chars in case</think>is truncated
                    safe = len(text) - 8
                    if safe > 0:
                        chunk = OpenAIResponse(
                            id=msg_id,
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=model,
                            choices=[OpenAIChoice(index=0, delta=OpenAIDelta(reasoning=text[:safe]))]
                        )
                        yield f"data: {json.dumps(chunk.dict())}\n\n"
                        text = text[safe:]
                    break

            buffer = text

        # Send remaining content
        if buffer:
            if in_think:
                chunk = OpenAIResponse(
                    id=msg_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[OpenAIChoice(index=0, delta=OpenAIDelta(reasoning=buffer))]
                )
            else:
                chunk = OpenAIResponse(
                    id=msg_id,
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=model,
                    choices=[OpenAIChoice(index=0, delta=OpenAIDelta(content=buffer))]
                )
            yield f"data: {json.dumps(chunk.dict())}\n\n"

        # Send end marker
        final_chunk = OpenAIResponse(
            id=msg_id,
            object="chat.completion.chunk",
            created=int(time.time()),
            model=model,
            choices=[OpenAIChoice(index=0, delta=OpenAIDelta(), finish_reason="stop")]
        )
        yield f"data: {json.dumps(final_chunk.dict())}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        error_chunk = {"error": {"message": str(e)}}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@router.get("/api/config")
async def get_config():
    """Get config"""
    return config_manager.get_config()


@router.post("/api/config")
async def update_config(request: Request):
    """Update config"""
    try:
        new_config = await request.json()
        config_manager.update_config(new_config)
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": "invalid"})


@router.post("/api/parse-curl")
async def parse_curl_command(request: ParseCurlRequest):
    """Parse cURL command"""
    account = parse_curl(request.curl)
    if not account:
        raise HTTPException(status_code=400, detail={"error": "parse failed"})
    return account.to_dict()


@router.post("/api/test-account")
async def test_account(request: TestAccountRequest):
    """Test account validity"""
    try:
        account = MimoAccount(
            service_token=request.service_token,
            user_id=request.user_id,
            xiaomichatbot_ph=request.xiaomichatbot_ph
        )

        client = MimoClient(account)
        content, _, _ = await client.call_api("hi", False)

        return {"success": True, "response": content}
    except Exception as e:
        return {"success": False, "error": str(e)}
