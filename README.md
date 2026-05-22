# Mimo2API Python

Converts Xiaomi MiMo AI into an OpenAI-compatible API with deep thinking and tool calling support.

## Features

- **OpenAI Compatible**: Fully compatible with OpenAI API format
- **Tool Calling**: OpenAI-compatible function/tool calling via prompt engineering (works with Cursor, Claude Code, Aider)
- **Deep Thinking**: Supports `reasoning_effort` parameter to enable reasoning mode
- **Streaming**: SSE real-time streaming responses
- **Account Rotation**: Multi-account load balancing
- **Web Admin**: Built-in admin UI for easy configuration
- **Async**: High-performance async implementation based on FastAPI
- **Auto Docs**: Auto-generated API docs at `/docs`

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Fix DNS (required)

MiMo's API domain resolves to `127.0.0.1`. Add a hosts entry:

```bash
echo "202.69.4.22 aistudio.xiaomimimo.com" | sudo tee -a /etc/hosts
```

### 3. Start the server

```bash
python3 main.py
```

The server starts at `http://localhost:8080`. Change the port with:

```bash
PORT=3000 python3 main.py
```

### 4. Configure an account

Open `http://localhost:8080` in your browser and follow these steps:

1. **Get credentials**:
   - Log in to [aistudio.xiaomimimo.com](https://aistudio.xiaomimimo.com)
   - Open browser developer tools (F12) -> Network tab
   - Send a message and find the `chat` request
   - Right-click -> Copy as cURL
   - Paste it into the admin UI

2. **Configure API Keys**:
   - Set custom API keys in the admin UI (comma-separated)
   - Default is `sk-default`

## Usage

### Basic request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-default" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mimo-v2.5-pro",
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

### Deep thinking

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-default" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mimo-v2.5-pro",
    "messages": [
      {"role": "user", "content": "Explain quantum entanglement"}
    ],
    "reasoning_effort": "medium"
  }'
```

### Streaming

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-default" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mimo-v2.5-pro",
    "messages": [
      {"role": "user", "content": "Write a poem"}
    ],
    "stream": true
  }'
```

### Tool calling

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-default" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mimo-v2.5-pro",
    "messages": [
      {"role": "user", "content": "List files in /tmp using bash"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "bash",
          "description": "Run a bash command",
          "parameters": {
            "type": "object",
            "properties": {
              "command": {
                "type": "string",
                "description": "The command to execute"
              }
            },
            "required": ["command"]
          }
        }
      }
    ]
  }'
```

## API Endpoints

| Endpoint | Method | Description | Auth |
|----------|--------|-------------|------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat endpoint | Bearer Token |
| `/api/config` | GET/POST | Get/update configuration | None |
| `/api/parse-curl` | POST | Parse cURL command to extract credentials | None |
| `/api/test-account` | POST | Test MiMo account validity | None |
| `/` | GET | Admin UI | None |
| `/docs` | GET | API documentation (Swagger UI) | None |

## Configuration

The `config.json` file is auto-created in the working directory:

```json
{
  "api_keys": "sk-default,sk-custom",
  "mimo_accounts": [
    {
      "service_token": "your_service_token",
      "user_id": "123456",
      "xiaomichatbot_ph": "your_xiaomichatbot_ph"
    }
  ]
}
```

## Environment Variables

- `PORT`: Server port (default: 8080)

```bash
PORT=3000 python3 main.py
```

## Using with Cursor / Claude Code / Aider

Point your tool to:

- **Base URL**: `http://localhost:8080/v1`
- **API Key**: `sk-default`
- **Model**: `mimo-v2.5-pro`

## Project Structure

```
MiMo2API/
├── app/
│   ├── __init__.py          # Package init
│   ├── config.py            # Configuration management
│   ├── mimo_client.py       # MiMo API client
│   ├── models.py            # OpenAI data models
│   ├── routes.py            # API routes
│   └── utils.py             # Utility functions + tool calling
├── web/
│   └── index.html           # Admin UI
├── main.py                  # Main entry point
├── requirements.txt         # Dependencies
├── test_tool_calling.py     # Tool calling test suite
└── README.md
```

## Tech Stack

- **Web Framework**: FastAPI 0.115.5
- **ASGI Server**: Uvicorn 0.32.1
- **HTTP Client**: httpx 0.27.2
- **Data Validation**: Pydantic 2.10.3

## Deep Thinking Mode

Enable deep thinking by setting the `reasoning_effort` parameter:

- `low`: Light reasoning
- `medium`: Moderate reasoning
- `high`: Deep reasoning

**Streaming format**:
```json
{"choices":[{"delta":{"reasoning":"Thinking content..."}}]}
{"choices":[{"delta":{"content":"Response content..."}}]}
```

**Non-streaming format**:
```json
{
  "choices": [{
    "message": {
      "content": "<think>Thinking content</think>\nResponse content"
    }
  }]
}
```

## Tool Calling

MiMo has no native tool calling support. This proxy implements it via prompt engineering:

- Tool definitions are injected into the system prompt
- The model responds with `<tool_call>` XML tags
- The proxy parses these into OpenAI-compatible `tool_calls` responses
- `finish_reason` is set to `"tool_calls"` when tools are invoked
- Multi-turn tool conversations are fully supported

**Limitations**: Since tool calling is prompt-engineered, the model may occasionally produce malformed tool calls or ignore tool instructions. `tool_choice="auto"` works best.

## Running Tests

```bash
python3 test_tool_calling.py
```

15 tests covering: basic tool calls, streaming, multi-turn, tool selection, tool_choice modes, response structure, auth.

## Development

```bash
# Auto-reload mode
uvicorn main:app --reload --port 8080
```

### API Documentation

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## Troubleshooting

### 1. Port in use

```bash
PORT=3000 python3 main.py
```

### 2. Invalid account

- Use the "Test" button in the admin UI to verify your account
- Re-capture the latest cURL command from the browser

### 3. DNS resolution error (connection refused)

Add the hosts entry:
```bash
echo "202.69.4.22 aistudio.xiaomimimo.com" | sudo tee -a /etc/hosts
```

### 4. Dependencies install failed

```bash
pip install -r requirements.txt --break-system-packages
```

## License

Based on the original Go version [mimo2api](https://github.com/leookun/mimo2api).

## Contributing

Issues and pull requests are welcome!

## Links

- [Original Go version](https://github.com/leookun/mimo2api)
- [Xiaomi MiMo AI](https://aistudio.xiaomimimo.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
