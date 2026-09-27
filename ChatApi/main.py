import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from trading_agent import prompt_model, stream_response

from fastapi.middleware.cors import CORSMiddleware

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from sqlmodel import Field, Session, SQLModel, create_engine, select

from config import settings

class ChatRequest(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    tool_model: str = Field(index=True)
    chat_model: str = Field(index=True)
    prompt: str = Field(index=True)

engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    create_db_and_tables()
    yield

app = FastAPI(lifespan=lifespan)

# Add CORS middleware - IMPORTANT!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/agent/trading/chat")
async def trading_agent_chat(request: ChatRequest) -> dict:
    return prompt_model(request.prompt, request.tool_model, request.chat_model)

@app.post("/agent/trading/chat/stream")
async def trading_agent_chat_stream(request: ChatRequest) -> StreamingResponse:
    with Session(engine) as session:
        session.add(request)
        session.commit()
        session.refresh(request) 
        return StreamingResponse(
            stream_response(request.prompt, request.tool_model, request.chat_model),
            media_type="text/plain"
        )

@app.get("/agent/trading/chats", response_model=Sequence[ChatRequest])
async def get_trading_agent_chats() -> Sequence[ChatRequest]:
    with Session(engine) as session:
        statement = select(ChatRequest)
        results = session.exec(statement)
        return results.all()