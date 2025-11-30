from typing import Union

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from trading_agent import prompt_model, stream_response

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware - IMPORTANT!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    tool_model: str
    chat_model: str
    prompt: str

@app.post("/agent/trading/chat")
async def trading_agent_chat(request: ChatRequest) -> dict:
    return await prompt_model(request.prompt, request.tool_model, request.chat_model)

@app.post("/agent/trading/chat/stream")
async def trading_agent_chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_response(request.prompt, request.tool_model, request.chat_model),
        media_type="text/plain"
    )