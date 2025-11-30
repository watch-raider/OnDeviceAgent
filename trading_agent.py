import os
import json

from langchain_ollama import ChatOllama

from langchain.messages import AIMessage

import yfinance as yf

import finance_tools as ft

tool_mapping = {
        "get_balance_sheet": ft.get_balance_sheet,
        "get_dividends": ft.get_dividends,
        "get_key_financial_metrics": ft.get_key_financial_metrics,
        "get_latest_news": ft.get_latest_news
}

def initialise_models(tool_model, chat_model, tools: list) -> dict:
    n_cores = os.cpu_count()
    model_dict = {}

    model_dict["tool_model"] = ChatOllama(
        model=tool_model,
        num_thread=n_cores,
        temperature=0.0
    ).bind_tools(tools)

    model_dict["chat_model"] = ChatOllama(
        model=chat_model,
        num_thread=n_cores,
        temperature=0.5
    )

    return model_dict

def initialise_chat(user_prompt: str) -> list[dict]:
    chat = [
        {
            "role": "system",
            "content": """You are a financial analysis assistant who has access to various tools for retrieving stock market and financial data about specific stocks.
            
            Use these tools to assist with user queries about stock performance, historical data and financial data.
            """
        },
        {
            "role": "user", 
            "content": user_prompt 
        }
    ]
    return chat

def execute_tool(tool_call: dict) -> dict:
    try:
        tool_response = tool_mapping[tool_call["name"]].invoke(tool_call["args"])
        # tool_func = tool_mapping[result.tool_calls[0]["name"]]
        # result = tool_func(**result.tool_calls[0]["args"])
    except Exception as e:
        result = {"error": str(e)}
        print(f"Error executing tool {tool_call['name']}: {e}")
    tool_response

    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "name": tool_call["name"],
        "content": tool_response
    }

async def prompt_model(prompt: str, tool_model: str, chat_model: str) -> dict:
    model_dict = initialise_models(
        tool_model, chat_model, 
        [ft.get_key_financial_metrics, ft.get_balance_sheet, ft.get_dividends, ft.get_latest_news]
    )

    chat = initialise_chat(prompt)
    tool_calls = []

    result = await model_dict["tool_model"].ainvoke(chat)
    if isinstance(result, AIMessage) and result.tool_calls:
        print(result.tool_calls)
        for tool_call in result.tool_calls:
            tool_result = execute_tool(tool_call)
            chat.append(tool_result)
            tool_calls.append({
                "name": tool_call["name"],
                "args": tool_call["args"]
            })

    if tool_calls:
        result = await model_dict["chat_model"].ainvoke(chat)
    
    return {
        "response": result.content,
        "tool_calls": tool_calls
    }

def stream_response(prompt: str, tool_model: str, chat_model: str):
    chat = initialise_chat(prompt)
    
    # Get tool selection model based on tool_model parameter
    model_dict = initialise_models(
        tool_model, chat_model, 
        [ft.get_key_financial_metrics, ft.get_balance_sheet, ft.get_dividends, ft.get_latest_news]
    )
    
    # Tool selection phase
    result = model_dict["tool_model"].invoke(chat)
    
    if isinstance(result, AIMessage) and result.tool_calls:
        for tool_call in result.tool_calls:
            yield f"data: {json.dumps({'type': 'tool', 'name': tool_call['name'], 'args': tool_call['args']})}\n\n"
            tool_result = execute_tool(tool_call)
            chat.append(tool_result)

    
    # Stream the final response
    response = model_dict["chat_model"].stream(chat)
    for chunk in response:
        yield f"data: {json.dumps({'type': 'text', 'content': chunk.content})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"
    
if __name__ == "__main__":
    model_dict = initialise_models(
        "ollama-350m", "ollama-1b",
        [ft.get_key_financial_metrics, ft.get_balance_sheet, ft.get_dividends, ft.get_latest_news]
    )
    prompt = input("Hi! I am your financial analysis assistant. How can I help you today?\n")
    while True:
        chat = initialise_chat(prompt)

        result = model_dict["ollama-350m"].invoke(chat)
        if isinstance(result, AIMessage) and result.tool_calls:
            print(result.tool_calls)
            for tool_call in result.tool_calls:
                tool_result = execute_tool(tool_call)
                chat.append(tool_result)

        result = model_dict["ollama-1b"].invoke(chat)
        print(result.content)
        prompt = input("\nDo you have any other questions? \n")