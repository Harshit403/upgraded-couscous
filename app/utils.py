"""Utility functions"""

import re
from typing import Optional
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
