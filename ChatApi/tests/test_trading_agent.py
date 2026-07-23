import sys
from pathlib import Path
import re

# Add parent directory (OnDeviceAgent) to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
import time
import yfinance as yf
from ChatApi.trading_agent import prompt_model, initialise_agent
import pytest


@pytest.fixture
def timer():
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"\nExecution time: {end - start:.4f} seconds")

# --------------------------------------------------------------
# Tests for Pricing
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("What is the current price of Nvidia", "granite4:350m", id="350m"),
        pytest.param("What is the current price of Nvidia", "granite4:1b", id="1b"),
        pytest.param("What is the current price of Nvidia", "granite4:3b", id="3b"),
    ]
)
def test_get_current_price(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("NVDA")
    y_price = dat.get_info()["currentPrice"]
    y = str(y_price)
    y = y.split(".")[0]  # match integer part only

    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected Price: {y}")

    assert y in x


@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("What was the price of Microsoft at close on the 31st October 2025", "granite4:350m", id="350m"),
        pytest.param("What was the price of Microsoft at close on the 31st October 2025", "granite4:1b", id="1b"),
        pytest.param("What was the price of Microsoft at close on the 31st October 2025", "granite4:3b", id="3b"),
    ]
)
def test_get_historic_price(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("MSFT")
    y_price = dat.history(period="1d", start="2025-10-31")["Close"].item()
    y = str(y_price)
    y = y.split(".")[0]  # match integer part only

    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected Price: {y}")

    assert y in x


@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("What was the highest price of Microsoft between 2025-10-30 and 2025-11-05", "granite4:350m", id="350m"),
        pytest.param("What was the highest price of Microsoft between 2025-10-30 and 2025-11-05", "granite4:1b", id="1b"),
        pytest.param("What was the highest price of Microsoft between 2025-10-30 and 2025-11-05", "granite4:3b", id="3b"),
    ]
)
def test_historical_high_range(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("MSFT")
    hist = dat.history(start="2025-10-30", end="2025-11-05")
    if hist.empty:
        pytest.skip("No historical data for MSFT in the given range")

    expected_val = hist['High'].max()

    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected High Price: {expected_val}")

    assert number_appears_in_text(target_number=expected_val, text=x, tolerance=0.1)

# --------------------------------------------------------------
# Tests for general tool calling and data retrieval
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("Can you tell me the market capilisation of AMD", "granite4:350m", id="350m"),
        pytest.param("Can you tell me the market capilisation of AMD", "granite4:1b", id="1b"),
        pytest.param("Can you tell me the market capilisation of AMD", "granite4:3b", id="3b"),
    ]
)
def test_get_market_cap(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("AMD")
    y_market_cap = dat.get_info()["marketCap"]

    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected Market Cap: {y_market_cap}")

    assert number_appears_in_text(target_number=y_market_cap, text=x, tolerance=0.1)


@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("Please use your tools to fetch the current price of Nvidia", "granite4:350m", id="350m"),
        pytest.param("Please use your tools to fetch the current price of Nvidia", "granite4:1b", id="1b"),
        pytest.param("Please use your tools to fetch the current price of Nvidia", "granite4:3b", id="3b"),
    ]
)
def test_tool_invokes_tools(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    tool_calls = response.get("tool_calls", [])

    assert len(tool_calls) > 0


@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("Give me the latest news for Tesla", "granite4:350m", id="350m"),
        pytest.param("Give me the latest news for Tesla", "granite4:1b", id="1b"),
        pytest.param("Give me the latest news for Tesla", "granite4:3b", id="3b"),
    ]
)
def test_get_latest_news(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("TSLA")
    news_list = dat.get_news()
    if not news_list:
        pytest.skip("No news available for TSLA")

    first_title = news_list[0].get('content', {}).get('title') or ""
    # pick a substantive word from the title for matching
    words = [w for w in re.findall(r"\w{5,}", first_title)]
    if not words:
        pytest.skip("No suitable title word to match")

    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected News Title Word(s): {words}")

    assert any(word in x for word in words)


@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("What dividends did Coca-Cola pay in the last year", "granite4:350m", id="350m"),
        pytest.param("What dividends did Coca-Cola pay in the last year", "granite4:1b", id="1b"),
        pytest.param("What dividends did Coca-Cola pay in the last year", "granite4:3b", id="3b"),
    ]
)
def test_get_dividends(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("KO")
    divs = dat.get_dividends(period="1y")
    if divs.empty:
        pytest.skip("No dividends available for KO in the given period")

    last_amt = float(divs.iloc[-1])

    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected Dividend Amount: {last_amt}")

    assert number_appears_in_text(target_number=last_amt, text=x, tolerance=0.1)

# --------------------------------------------------------------
# Tests for Balance Sheet metrics 
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("Please fetch the balance sheet for Microsoft and tell me the Total Assets", "granite4:350m", id="350m"),
        pytest.param("Please fetch the balance sheet for Microsoft and tell me the Total Assets", "granite4:1b", id="1b"),
        pytest.param("Please fetch the balance sheet for Microsoft and tell me the Total Assets", "granite4:3b", id="3b"),
    ]
)
def test_get_balance_sheet(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("MSFT")
    try:
        bs = dat.get_balance_sheet()
    except Exception:
        pytest.skip("Could not retrieve balance sheet from yfinance")

    if bs.empty:
        pytest.skip("No balance sheet available for MSFT")

    expected_val = bs.loc["TotalAssets"].iloc[0] if "TotalAssets" in bs.index else bs.iloc[0, 0]
    
    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected Total Assets: {expected_val}")

    assert number_appears_in_text(target_number=expected_val, text=x, tolerance=0.1), f"Expected balance-sheet value {expected_val} not found in response: {x}"


@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("Can you calculate AMD's current ratio for 2025 using the balance sheet?", "granite4:350m", id="350m"),
        pytest.param("Can you calculate AMD's current ratio for 2025 using the balance sheet?", "granite4:1b", id="1b"),
        pytest.param("Can you calculate AMD's current ratio for 2025 using the balance sheet?", "granite4:3b", id="3b"),
    ]
)
def test_calculate_current_ratio(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("AMD")
    balance_sheet = dat.get_balance_sheet()

    cols_2025 = [col for col in balance_sheet.columns if col.year == 2025]
    current_assets = balance_sheet.loc["CurrentAssets", max(cols_2025)].item()
    current_liabilities = balance_sheet.loc["CurrentLiabilities", max(cols_2025)].item()
    expected_ratio = current_assets / current_liabilities

    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected Current Ratio: {expected_ratio}")

    assert number_appears_in_text(target_number=expected_ratio, text=x), f"Expected current ratio {expected_ratio} not found in response: {x}"


@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("Can you calculate ASML's debt to equity for 2024 using the balance sheet?", "granite4:350m", id="350m"),
        pytest.param("Can you calculate ASML's debt to equity for 2024 using the balance sheet?", "granite4:1b", id="1b"),
        pytest.param("Can you calculate ASML's debt to equity for 2024 using the balance sheet?", "granite4:3b", id="3b"),
    ]
)
def test_calculate_debt_to_equity(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("ASML")
    balance_sheet = dat.get_balance_sheet()

    cols_2024 = [col for col in balance_sheet.columns if col.year == 2024]
    total_debt = balance_sheet.loc["TotalDebt", max(cols_2024)].item()
    total_equity = balance_sheet.loc["StockholdersEquity", max(cols_2024)].item()
    expected_ratio = total_debt / total_equity

    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected Debt-to-Equity Ratio: {expected_ratio}")

    assert number_appears_in_text(target_number=expected_ratio, text=x), f"Expected debt-to-equity ratio {expected_ratio} not found in response: {x}"

# --------------------------------------------------------------
# Tests for Income Statement 
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("Please fetch the income statement for Microsoft and tell me Total Revenue", "granite4:350m", id="350m"),
        pytest.param("Please fetch the income statement for Microsoft and tell me Total Revenue", "granite4:1b", id="1b"),
        pytest.param("Please fetch the income statement for Microsoft and tell me Total Revenue", "granite4:3b", id="3b"),
    ]
)
def test_get_income_statement(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("MSFT")
    try:
        inc = dat.get_income_stmt()
    except Exception:
        pytest.skip("Could not retrieve income statement from yfinance")

    if inc.empty:
        pytest.skip("No income statement available for MSFT")

    # Prefer 'Total Revenue' label when available
    expected_val = inc.loc["TotalRevenue"].iloc[0] if "TotalRevenue" in inc.index else inc.iloc[0, 0]
    
    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected Total Revenue: {expected_val}")
    
    assert number_appears_in_text(target_number=expected_val, text=x, tolerance=0.1), f"Expected revenue {expected_val} not found in response: {x}"

@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("Please fetch the income statement for Meta and tell me the latest Gross Profit Margin", "granite4:350m", id="350m"),
        pytest.param("Please fetch the income statement for Meta and tell me the latest Gross Profit Margin", "granite4:1b", id="1b"),
        pytest.param("Please fetch the income statement for Meta and tell me the latest Gross Profit Margin", "granite4:3b", id="3b"),
    ]
)
def test_gross_profit_margin(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("Meta")
    try:
        inc = dat.get_income_stmt()
    except Exception:
        pytest.skip("Could not retrieve income statement from yfinance")

    if inc.empty:
        pytest.skip("No income statement available for Meta")

    # Prefer 'Total Revenue' label when available
    expected_val = (inc.loc["GrossProfit"].iloc[0] / inc.loc["TotalRevenue"].iloc[0]) * 100
    
    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected Gross Profit Margin: {expected_val}")
    
    assert number_appears_in_text(target_number=expected_val, text=x, tolerance=0.1), f"Expected gross profit margin {expected_val} not found in response: {x}"

# --------------------------------------------------------------
# Tests for Cash Flow Statement
# --------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt, model_name",
    [
        pytest.param("Please fetch the cash flow statement for Microsoft and tell me Capital Expenditure", "granite4:350m", id="350m"),
        pytest.param("Please fetch the cash flow statement for Microsoft and tell me Capital Expenditure", "granite4:1b", id="1b"),
        pytest.param("Please fetch the cash flow statement for Microsoft and tell me Capital Expenditure", "granite4:3b", id="3b"),
    ]
)
def test_get_cash_flow(prompt:str, model_name:str, timer):
    agent = initialise_agent(model_name)
    response = prompt_model(prompt, agent)
    x = response["response"]

    dat = yf.Ticker("MSFT")
    try:
        cf = dat.get_cashflow()
    except Exception:
        pytest.skip("Could not retrieve cash flow statement from yfinance")

    if cf.empty:
        pytest.skip("No cash flow data available for MSFT")

    # Try to locate a sensible operating cash flow label
    expected_val = cf.loc["CapitalExpenditure"].iloc[0] if "CapitalExpenditure" in cf.index else cf.iloc[0, 0]

    print("Tool calls: ", response["tool_calls"])
    print(f"Response: {x}, Expected Capital Expenditure: {expected_val}")
    
    assert number_appears_in_text(target_number=expected_val, text=x, tolerance=0.1), f"Expected cash flow value {expected_val} not found in response: {x}"

# --------------------------------------------------------------
# Helper functions for number extraction and matching
# --------------------------------------------------------------

def number_appears_in_text(target_number, text, tolerance=0.01):
    """
    Check if a number appears in text, handling various formats.
    
    Args:
        target_number: The float to search for
        text: The text to search in
        tolerance: Acceptable relative difference (default 1%)
                  For numbers < 1, uses absolute tolerance instead
    
    Returns:
        bool: True if number found within tolerance
    """
    
    # Multiplier words mapping
    multipliers = {
        'thousand': 1_000,
        'k': 1_000,
        'million': 1_000_000,
        'm': 1_000_000,
        'billion': 1_000_000_000,
        'b': 1_000_000_000,
        'trillion': 1_000_000_000_000,
        't': 1_000_000_000_000,
    }
    
    # Pattern to match numbers with optional negative sign, currency symbols, commas, decimals, and multipliers
    # Matches: -$64,551,000,000 or -64551000000 or $3.5m or -3.5 million
    pattern = r'[-−]?\s*[$€£¥]?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(thousand|million|billion|trillion|[kmbt])?\b'
    
    matches = re.finditer(pattern, text, re.IGNORECASE)
    
    for match in matches:
        # Get the full match to check for negative sign
        full_match = match.group(0)
        is_negative = full_match.strip().startswith('-') or full_match.strip().startswith('−')
        
        num_str = match.group(1).replace(',', '')
        multiplier_str = match.group(2)
        
        try:
            value = float(num_str)
            
            # Apply negative sign if present
            if is_negative:
                value = -value
            
            # Apply multiplier if present
            if multiplier_str:
                multiplier_key = multiplier_str.lower()
                if multiplier_key in multipliers:
                    value *= multipliers[multiplier_key]
            
            # Use absolute tolerance for small numbers, relative for large
            if abs(target_number) < 1:
                # For ratios and small numbers, use absolute tolerance
                if abs(value - target_number) <= tolerance:
                    return True
            else:
                # For large numbers, use relative tolerance
                if abs(value - target_number) <= abs(target_number * tolerance):
                    return True
        
        except ValueError:
            continue
    
    return False

if __name__ == "__main__":
    pass
    # test_get_current_price("What is the current price of Nvidia", "granite4:350m", "granite4:350m")
    # test_get_historic_price("What was the price of Microsoft at close on the 31st October 2025", "granite4:350m", "granite4:1b")
    test_get_balance_sheet("Please fetch the balance sheet for Microsoft and tell me the Total Assets", "granite4:1b", "granite4:1b")