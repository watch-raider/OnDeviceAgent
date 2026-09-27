import sys
from pathlib import Path
import os

# Add parent directory (OnDeviceAgent) to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from ChatApi.trading_agent import prompt_model, stream_response, model_parameter

from typing import Annotated

app = FastAPI()


# Add CORS middleware - IMPORTANT!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CommonsDep = Annotated[dict, Depends(model_parameter)]

@app.get("/agent/trading/chat")
async def trading_agent_chat(prompt: str, agent: CommonsDep):
    return prompt_model(prompt, agent)

@app.get("/agent/trading/chat/stream")
async def trading_agent_chat_stream(prompt: str, agent: CommonsDep) -> StreamingResponse:
    return StreamingResponse(
        stream_response(prompt, agent),
        media_type="text/plain"
    )