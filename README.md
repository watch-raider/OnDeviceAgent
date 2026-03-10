# OnDeviceAgent

A local AI financial assistant built with Langchain and small open-source models that run entirely on your machine. The project evaluates how capable small language models are at tool-use tasks for financial analysis, with the goal of avoiding frontier models where user data may not be under their control.

## Motivation

Frontier models (GPT-4, Claude, Gemini) require sending user data to external servers, raising privacy concerns. This project explores whether small, locally-running open-source models can effectively serve as capable AI assistants for a confined set of tasks.

## Architecture

- **Langchain** - Agent framework for tool orchestration
- **Ollama** - Run open-source models locally (LLM inference)
- **LangGraph** - Agent state management with checkpointing
- **FastAPI** - REST API server for the agent
- **yfinance** - Stock market data API

## Available Tools

The agent has access to the following financial data tools:

| Tool | Description |
|------|-------------|
| `get_historical_data` | Historical stock prices |
| `get_key_financial_metrics` | Key metrics (P/E, market cap, revenue, etc.) |
| `get_balance_sheet` | Company balance sheet data |
| `get_income_statement` | Revenue, expenses, profits |
| `get_cash_flow_statement` | Cash flow from operations |
| `get_dividends` | Dividend history |
| `get_latest_news` | Recent news for a ticker |

## Models Tested

Evaluates IBM Granite models of varying sizes:
- `granite4:350m` (350M parameters)
- `granite4:1b` (1B parameters)  
- `granite4:3b` (3B parameters)

## Running the Agent

### Prerequisites

1. Install Ollama and pull the desired model:
```bash
ollama pull granite4:1b
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Start the API and Web UI

```bash
./startup.sh
```

This script:
- Launches the Ollama app
- Activates the virtual environment
- Opens the web chat UI in your browser
- Starts the FastAPI server
