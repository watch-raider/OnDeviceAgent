import pandas as pd
import json

import yfinance as yf

from langchain.tools import tool

@tool
def get_historical_data(ticker: str, period: str = "1d", start: str = None) -> str:
    """Get historical market price data for a given ticker symbol.
    
    Args:
        ticker (str): The ticker symbol of the company e.g. "MSFT".
        period (str): The period over which to fetch data. Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max. Default: 1d. Can combine with start/end e.g. end = start + period
        start (str): The start date for fetching historical data in 'YYYY-MM-DD' format. Optional.

    Returns:
        str: Historical market data as a string in tabular format.
    """
    dat = yf.Ticker(ticker)
    hist = dat.history(period=period, start=start)
    return hist.to_csv(index=True)

@tool
def get_latest_news(ticker: str) -> str:
    """Get the latest news articles for a given ticker symbol.

    Args:
        ticker (str): The ticker symbol of the company e.g. "MSFT".
    """

    dat = yf.Ticker(ticker)
    news_list = dat.get_news()
    
    extracted_news = []
    
    for article in news_list[:5]:  # Limit to first 5 articles
        content = article.get('content', {})
        
        key_info = {
            'id': content.get('id'),
            'title': content.get('title'),
            'summary': content.get('summary'),
            'pubDate': content.get('pubDate'),
            'provider': content.get('provider', {}).get('displayName'),
            'contentType': content.get('contentType'),
        }
        
        # Remove None values
        key_info = {k: v for k, v in key_info.items() if v is not None}
        extracted_news.append(key_info)

    json_output = json.dumps(extracted_news, indent=2)
    return json_output

@tool
def get_key_financial_metrics(ticker: str) -> str:
    """Get most important financial metrics.
    e.g. current price, market cap, P/E ratios, revenue, earnings, margins, cash flow, dividends, analyst target price.
    
    Args:
        ticker (str): The ticker symbol of the company e.g. "MSFT".
    """
    dat = yf.Ticker(ticker)
    full_data = dat.get_info()

    # Define the keys we want to get
    important_keys = [
        # Price Data
        "currentPrice", 
        # Valuation
        "marketCap",
        # Valuation Ratios
        "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
        # Revenue & Earnings
        "totalRevenue", "revenueGrowth", "earningsGrowth"
        # Margins
        "profitMargins", "grossMargins", "operatingMargins", "ebitdaMargins",  
        # Balance Sheet
        "totalCash", "totalDebt", "debtToEquity", "currentRatio", "quickRatio",
        # Cash Flow
        "operatingCashflow", "freeCashflow",
        # Dividends
        "dividendRate", "dividendYield", "payoutRatio",
        # Analyst Data
        "targetMeanPrice"
    ]
    
    # Extract only the keys that exist in the full data
    extracted_data = {
        key: full_data[key] 
        for key in important_keys 
        if key in full_data
    }
    
    json_output = json.dumps(extracted_data, indent=2)
    return json_output

@tool
def get_balance_sheet(ticker: str) -> str:
    """Get the balance sheet of a company given its ticker symbol. 
    The balance sheet provides a snapshot of the company's assets, liabilities, and shareholders' equity at a specific point in time.
    
    Args:
        ticker (str): The ticker symbol of the company e.g. "MSFT".

    Returns:
        str: The balance sheet of the company as a string in tabular format.
    """
    dat = yf.Ticker(ticker)
    return dat.get_balance_sheet().to_csv(index=True)

@tool
def get_income_statement(ticker: str) -> str:
    """Get the income statement of a company given its ticker symbol.
    The income statement provides insights into the company's revenues, expenses, and profits over a specific period.
    
    Args:
        ticker (str): The ticker symbol of the company e.g. "MSFT".

    Returns:
        str: The income statement of the company as a string in tabular format.
    """
    dat = yf.Ticker(ticker)
    return dat.get_income_stmt().to_csv(index=True)

@tool
def get_cash_flow_statement(ticker: str) -> str:
    """Get the cash flow statement of a company given its ticker symbol. 
    The cash flow statement provides insights into the cash inflows and outflows from operating, investing, and financing activities.
    
    Args:
        ticker (str): The ticker symbol of the company e.g. "MSFT".

    Returns:
        str: The cash flow statement of the company as a string in tabular format.
    """
    dat = yf.Ticker(ticker)
    return dat.get_cashflow().to_csv(index=True)

@tool
def get_dividends(ticker: str, time_period: str = "1mo") -> str:
    """Get the dividends of a company given its ticker symbol.
    
    Args:
        ticker (str): The ticker symbol of the company e.g. "MSFT".
        time_period (str): How far back in the past from today to fetch dividends. Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max. Default: 1mo

    Returns:
        str: The dividends of the company as a string in tabular format.
    """
    dat = yf.Ticker(ticker)
    return dat.get_dividends(period=time_period).to_csv(index=True)