import sys
from pathlib import Path

# Add parent directory (OnDeviceAgent) to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json

from langchain_ollama import ChatOllama

from langchain.messages import HumanMessage, AIMessage, ToolMessage

import yfinance as yf

import ChatApi.finance_tools as ft

from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()

tool_mapping = {
        "get_historical_data": ft.get_historical_data,
        "get_balance_sheet": ft.get_balance_sheet,
        "get_dividends": ft.get_dividends,
        "get_key_financial_metrics": ft.get_key_financial_metrics,
        "get_latest_news": ft.get_latest_news,
        "get_income_statement": ft.get_income_statement,
        "get_cash_flow_statement": ft.get_cash_flow_statement
}

def prompt_model(prompt: str, agent) -> dict:
    messages = [
        HumanMessage(content=prompt)
    ]
    result = agent.invoke(
        {"messages": messages}, 
        {"configurable": {"thread_id": "1"}}
    )
    
    return {
        "response": result["messages"][-1].content,
        "tool_calls": [msg.tool_calls for msg in result["messages"] if isinstance(msg, AIMessage)]
    }

def stream_response(prompt: str, agent):
    messages = [
        HumanMessage(content=prompt)
    ] 

    for token, metadata in agent.stream({"messages": messages}, {"configurable": {"thread_id": "1"}}, stream_mode="messages"):
        if metadata['langgraph_node'] == "model":
            for content_block in token.content_blocks:
                if content_block["type"] == "text":
                    yield f"data: {json.dumps({'type': 'text', 'content': content_block['text']})}\n\n"
                elif content_block["type"] == "tool_call_chunk":
                    if content_block.get("name") and content_block.get("args"):
                        yield f"data: {json.dumps({'type': 'tool', 'name': content_block['name'], 'args': content_block['args']})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"