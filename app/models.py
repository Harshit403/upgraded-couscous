"""OpenAI API data models"""

from typing import List, Optional, Literal, Any, Dict
from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    """Function call in a tool call"""
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Tool call in a message"""
    id: str
    type: str = "function"
    function: FunctionCall


class OpenAIMessage(BaseModel):
    """OpenAI message"""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class FunctionDefinition(BaseModel):
    """Function definition in a tool"""
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    """Tool definition"""
    type: str = "function"
    function: FunctionDefinition


class OpenAIRequest(BaseModel):
    """OpenAI request"""
    model: str
    messages: List[OpenAIMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = Field(None, description="Reasoning effort level: low/medium/high")
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Any] = None


class OpenAIDelta(BaseModel):
    """OpenAI streaming response delta"""
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning: Optional[str] = Field(None, description="Reasoning content")
    tool_calls: Optional[List[ToolCall]] = None


class OpenAIChoice(BaseModel):
    """OpenAI choice"""
    index: int
    message: Optional[OpenAIMessage] = None
    delta: Optional[OpenAIDelta] = None
    finish_reason: Optional[str] = Field(None, description="stop, length, or tool_calls")


class OpenAIUsage(BaseModel):
    """OpenAI usage stats"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponse(BaseModel):
    """OpenAI response"""
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: Optional[OpenAIUsage] = None


class ParseCurlRequest(BaseModel):
    """Parse cURL request"""
    curl: str


class TestAccountRequest(BaseModel):
    """Test account request"""
    service_token: str
    user_id: str
    xiaomichatbot_ph: str
