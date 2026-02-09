import yfinance as yf
import pandas as pd
import numpy as np
import os
import torch
from datetime import datetime, timedelta

# --- Configuration ---
START_DATE = (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')
DATA_DIR = "./data"
PROCESSED_DIR = "./data/processed"

MACRO_TICKERS = {
    "Energy": "CL=F",
    "Rates": "^TNX",
    "Currency": "DX-Y.NYB",
    "Fear": "^VIX"
}

# --- Indicator Functions ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, slow=26, fast=12, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_volatility(series, window=5):
    # Log returns volatility
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window=window).std()

def rolling_zscore_normalize(df, window=60):
    """
    Apply Rolling Z-Score normalization to all columns.
    
    For each column: (x - rolling_mean) / rolling_std
    This ensures all features are scaled to ~N(0,1).
    
    Args:
        df: DataFrame with feature columns
        window: Rolling window size (default 60 days)
    
    Returns:
        Normalized DataFrame with same columns
    """
    normalized = pd.DataFrame(index=df.index)
    for col in df.columns:
        rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
        rolling_std = df[col].rolling(window=window, min_periods=1).std()
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-8)
        normalized[col] = (df[col] - rolling_mean) / rolling_std
    return normalized

# --- Main Execution ---
def fetch_sp500_tickers():
    print("Scraping S&P 500 tickers from Wikipedia...")
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    # Replace dots with dashes for Yahoo (e.g. BRK.B -> BRK-B)
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers

def process_ticker_data(ticker, is_macro=False):
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if df.empty:
            return None
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Basic Features
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Log_Vol'] = np.log(df['Volume'] + 1)
        
        # Technicals
        df['RSI_14'] = calculate_rsi(df['Close'])
        macd, _ = calculate_macd(df['Close'])
        df['MACD'] = macd
        df['Volatility_5'] = calculate_volatility(df['Close'])
        
        # Macro Specifics (Moving Averages)
        if is_macro:
            df['MA_50'] = df['Close'].rolling(window=50).mean()
            df['MA_200'] = df['Close'].rolling(window=200).mean()
            # Keep only Macro Features defined in schema
            cols = ['Close', 'Returns', 'MA_50', 'MA_200']
            df = df[cols]
        else:
            # Keep Stock Features defined in schema
            cols = ['Close', 'High', 'Low', 'Log_Vol', 'RSI_14', 'MACD', 'Volatility_5', 'Returns']
            df = df[cols]

        # Apply Rolling Z-Score Normalization
        # Critical: Neural networks require inputs scaled to ~N(0,1)
        df = rolling_zscore_normalize(df, window=60)
        
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 1. Fetch Macro Data
    print("Fetching Macro Nodes...")
    macro_data = {}
    for name, ticker in MACRO_TICKERS.items():
        df = process_ticker_data(ticker, is_macro=True)
        if df is not None:
            macro_data[name] = df
    
    print(f"Successfully fetched {len(macro_data)} macro nodes.")

    # 2. Fetch Stock Data
    tickers = fetch_sp500_tickers()
    print(f"Found {len(tickers)} S&P 500 tickers. Starting download (this may take time)...")
    
    stock_data = {}
    
    # For PoC, limit to first 50 to save time/bandwidth, remove slice [:] for full run
    for ticker in tickers[:50]: 
        df = process_ticker_data(ticker, is_macro=False)
        if df is not None and len(df) > 200:  # Ensure enough history
            stock_data[ticker] = df
            print(f"Processed {ticker} - {len(df)} rows")
    
    # 3. CRITICAL: Align all dataframes to common date index
    # Different tickers have different histories (VIX needs 200 days for MA_200)
    # This prevents tensor shape mismatches during training
    print("\nAligning all data to common date range...")
    
    # Find intersection of all date indices
    all_indices = [df.index for df in macro_data.values()] + [df.index for df in stock_data.values()]
    
    if not all_indices:
        print("ERROR: No valid data found!")
        return
    
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.intersection(idx)
    
    common_index = common_index.sort_values()
    print(f"Common date range: {common_index[0]} to {common_index[-1]} ({len(common_index)} trading days)")
    
    if len(common_index) < 200:
        print("WARNING: Common date range is very short. Consider using fewer macro indicators or longer history.")
    
    # 4. Align and save all data
    for name, df in macro_data.items():
        aligned_df = df.loc[common_index]
        aligned_df.to_csv(f"{PROCESSED_DIR}/macro_{name}.csv")
    
    valid_stocks = []
    for ticker, df in stock_data.items():
        aligned_df = df.loc[common_index]
        aligned_df.to_csv(f"{PROCESSED_DIR}/stock_{ticker}.csv")
        valid_stocks.append(ticker)
    
    # Save common index for reference
    pd.Series(common_index).to_csv(f"{PROCESSED_DIR}/common_index.txt", index=False, header=False)
    
    print(f"\nIngestion Complete. Processed {len(valid_stocks)} stocks and {len(macro_data)} macro nodes.")
    print(f"All data aligned to {len(common_index)} common trading days.")
    print(f"Data saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()