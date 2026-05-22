"""OpenAI API data models"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class OpenAIMessage(BaseModel):
    """OpenAI message"""
    role: str
    content: str


class OpenAIRequest(BaseModel):
    """OpenAI request"""
    model: str
    messages: List[OpenAIMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    reasoning_effort: Optional[str] = Field(None, description="Reasoning effort level: low/medium/high")


class OpenAIDelta(BaseModel):
    """OpenAI streaming response delta"""
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning: Optional[str] = Field(None, description="Reasoning content")


class OpenAIChoice(BaseModel):
    """OpenAI choice"""
    index: int
    message: Optional[OpenAIMessage] = None
    delta: Optional[OpenAIDelta] = None
    finish_reason: Optional[str] = None


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
