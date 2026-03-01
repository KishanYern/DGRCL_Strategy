import yfinance as yf
import pandas as pd
import numpy as np
import os
import torch
from datetime import datetime, timedelta

# --- Configuration ---
START_DATE = "2007-01-01"  # Full historical backtest window
END_DATE = datetime.now().strftime('%Y-%m-%d')
DATA_DIR = "./data"
PROCESSED_DIR = "./data/processed"
HISTORICAL_CSV = "./data/sp500_historical.csv"

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
        # .shift(1) prevents look-ahead: at time t, stats use [t-window, t-1]
        rolling_mean = df[col].rolling(window=window, min_periods=1).mean().shift(1)
        rolling_std = df[col].rolling(window=window, min_periods=1).std().shift(1)
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, 1e-8)
        normalized[col] = (df[col] - rolling_mean) / rolling_std
    return normalized


# =============================================================================
# HISTORICAL CONSTITUENT LOADING
# =============================================================================

def generate_mock_historical_constituents(csv_path: str):
    """
    Generate a realistic mock sp500_historical.csv for development.
    
    Produces quarterly entries from 2007-2025 with realistic churn:
    - Starts with ~490 tickers
    - ~3-7 adds/removes per quarter
    - ~600 total unique tickers across the full timeline
    """
    import random
    random.seed(42)
    
    # Pool of realistic ticker symbols (mix of current and historical S&P 500)
    ticker_pool = [
        # Technology
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
        "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "ACN", "IBM", "INTC",
        "AMD", "QCOM", "TXN", "NOW", "INTU", "AMAT", "MU", "LRCX",
        "ADI", "KLAC", "SNPS", "CDNS", "MCHP", "FTNT", "PANW", "CRWD",
        "NFLX", "PYPL", "SQ", "SHOP", "UBER", "ABNB", "SNAP", "PINS",
        # Healthcare
        "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
        "DHR", "BMY", "AMGN", "GILD", "MDT", "SYK", "ISRG", "REGN",
        "VRTX", "BSX", "ZBH", "BAX", "BDX", "EW", "CI", "HUM",
        "CVS", "MCK", "CAH", "ABC", "BIIB", "ILMN", "IQV", "DXCM",
        # Financials
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS",
        "SPGI", "BLK", "C", "AXP", "SCHW", "CB", "MMC", "PGR",
        "ICE", "CME", "AON", "MET", "AIG", "PRU", "TRV", "ALL",
        "AFL", "USB", "PNC", "TFC", "COF", "FIS", "FISV", "DFS",
        # Consumer Discretionary
        "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "MAR",
        "HLT", "ORLY", "AZO", "ROST", "DHI", "LEN", "PHM", "GM",
        "F", "APTV", "EBAY", "ETSY", "BBY", "DG", "DLTR", "KMX",
        # Industrials
        "GE", "CAT", "HON", "UNP", "UPS", "RTX", "BA", "LMT",
        "DE", "MMM", "GD", "NOC", "WM", "RSG", "CSX", "NSC",
        "EMR", "ETN", "ITW", "ROK", "PH", "CMI", "FDX", "DAL",
        "LUV", "AAL", "UAL", "XPO", "FAST", "PCAR", "IR", "DOV",
        # Consumer Staples
        "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ",
        "CL", "EL", "KMB", "GIS", "K", "SJM", "HSY", "HRL",
        "CPB", "MKC", "CHD", "CLX", "KR", "SYY", "ADM", "TSN",
        # Energy
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
        "PXD", "OXY", "HES", "DVN", "FANG", "HAL", "BKR", "KMI",
        "WMB", "OKE", "CTRA", "MRO", "APA", "TRGP",
        # Utilities
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL",
        "ED", "WEC", "DTE", "AWK", "ES", "PPL", "FE", "AEE",
        "CMS", "CNP", "EVRG", "ATO", "NI", "PNW", "LNT",
        # Real Estate
        "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "DLR",
        "WELL", "AVB", "EQR", "VTR", "ARE", "MAA", "ESS", "UDR",
        "HST", "KIM", "REG", "FRT", "BXP", "SLG",
        # Materials
        "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "VMC",
        "MLM", "DOW", "DD", "PPG", "CE", "ALB", "CF", "MOS",
        "IFF", "EMN", "IP", "PKG", "SEE", "AVY", "BLL",
        # Communication Services
        "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "ATVI",
        "WBD", "PARA", "FOXA", "FOX", "NWSA", "NWS", "OMC", "IPG",
        "TTWO", "LYV", "MTCH",
        # Historical tickers (removed from S&P 500 over time - no duplicates)
        "XRX", "HPQ", "TWX", "YHOO", "DELL",
        "MER", "LEH", "BSC", "WB", "FNM", "FRE",
        "CIT", "CBE", "EDS", "UST", "SGP", "WYE",
        "MBI", "ABK", "CFC", "JAVA", "SIRI", "TIE", "JNY",
        "DJ", "TE", "HPC", "ROH", "ASH", "EK", "PBI",
        "WIN", "JCP", "S", "SHLD", "ANR", "BEAM", "LO",
        "JOY", "ARG", "PLL", "DNR", "DO", "CHK", "FTR",
        "ENDP", "RAD", "BBBY", "KSS", "M", "GPS", "LUMN",
        "WYNN", "LB", "COTY", "HBI", "IVZ", "BEN", "FHN",
        "PBCT", "CMA", "ZION", "SIVB", "FRC",
        # More recent additions
        "PLTR", "DDOG", "ZS", "SNOW", "NET", "TEAM", "MDB",
        "COIN", "RIVN", "LCID", "CEG", "CARR", "OTIS", "CTVA",
        "TECH", "MRNA", "ENPH", "SEDG",
        "GNRC", "MPWR", "ON", "SMCI", "DECK", "PODD", "AXON",
    ]
    
    # Deduplicate (handles any remaining overlapping symbols)
    ticker_pool = sorted(list(set(ticker_pool)))
    
    # Start with ~490 tickers from the pool
    current_members = set(random.sample(ticker_pool, min(490, len(ticker_pool))))
    remaining_pool = set(ticker_pool) - current_members
    
    rows = []
    dates = pd.date_range("2007-01-01", "2025-12-31", freq="QS")
    
    for date in dates:
        # Record current snapshot
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "tickers": ",".join(sorted(current_members))
        })
        
        # Simulate quarterly churn: remove 3-7, add 3-7
        n_remove = random.randint(3, 7)
        n_add = random.randint(3, 7)
        
        removable = list(current_members)
        if len(removable) > n_remove:
            removed = set(random.sample(removable, n_remove))
            current_members -= removed
            remaining_pool |= removed
        
        addable = list(remaining_pool)
        if len(addable) > n_add:
            added = set(random.sample(addable, n_add))
            current_members |= added
            remaining_pool -= added
    
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Generated mock historical constituents: {csv_path}")
    print(f"  {len(dates)} quarterly snapshots, {len(ticker_pool)} total unique tickers")
    return df


def load_historical_constituents(csv_path: str = None):
    """
    Load point-in-time S&P 500 constituent data.
    
    Args:
        csv_path: Path to sp500_historical.csv (columns: date, tickers)
    
    Returns:
        constituent_df: DataFrame with 'date' and 'tickers' columns
        all_tickers: Sorted list of every unique ticker across the timeline
    """
    if csv_path is None:
        csv_path = HISTORICAL_CSV
    
    if not os.path.exists(csv_path):
        print(f"Historical constituent file not found at {csv_path}")
        print("Generating mock historical constituents for development...")
        generate_mock_historical_constituents(csv_path)
    
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract superset of all unique tickers
    all_tickers = set()
    for tickers_str in df['tickers']:
        all_tickers.update(tickers_str.split(','))
    
    all_tickers = sorted(all_tickers)
    
    print(f"Loaded historical constituents from {csv_path}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  {len(df)} snapshots, {len(all_tickers)} unique tickers (superset)")
    
    return df, all_tickers


# --- Main Execution ---

def process_ticker_data(ticker, is_macro=False):
    """
    Download and process a single ticker.
    
    CRITICAL: All indicators and normalization are computed on VALID data only.
    Zero-fill reindexing is NOT done here — it happens later in main() after
    all processing is complete (Process-First, Reindex-Last pattern).
    """
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if df.empty:
            return None
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Basic Features — computed on valid data only
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Log_Vol'] = np.log(df['Volume'] + 1)
        
        # Technicals — computed on valid data only
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

        # Apply Rolling Z-Score Normalization on VALID data only
        # Critical: This must happen BEFORE any zero-fill reindexing
        df = rolling_zscore_normalize(df, window=60)
        
        # Drop NaN rows from indicator warmup (RSI, MACD, rolling stats)
        df = df.dropna()
        return df
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None


def main():
    if os.path.exists(PROCESSED_DIR):
        import shutil
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 1. Load historical constituent data → extract superset
    constituent_df, all_tickers = load_historical_constituents()
    
    # 2. Fetch Macro Data
    print("\nFetching Macro Nodes...")
    macro_data = {}
    for name, ticker in MACRO_TICKERS.items():
        df = process_ticker_data(ticker, is_macro=True)
        if df is not None:
            macro_data[name] = df
    
    print(f"Successfully fetched {len(macro_data)} macro nodes.")

    # 3. Fetch Stock Data for entire superset
    print(f"\nDownloading data for {len(all_tickers)} tickers (full superset)...")
    print("This may take significant time for the first run.\n")
    
    stock_data = {}
    failed_tickers = []
    
    for i, ticker in enumerate(all_tickers):
        df = process_ticker_data(ticker, is_macro=False)
        if df is not None and len(df) > 0:
            stock_data[ticker] = df
        else:
            failed_tickers.append(ticker)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(all_tickers)} tickers "
                  f"({len(stock_data)} valid, {len(failed_tickers)} failed)")
    
    print(f"\nDownload complete: {len(stock_data)} valid stocks, "
          f"{len(failed_tickers)} failed/empty")
    
    if not stock_data:
        print("ERROR: No valid stock data found!")
        return
    
    # 4. Build UNION date index (instead of intersection)
    # This ensures we have dates for all tickers, even those with partial histories
    print("\nBuilding equity union date index...")
    
    # Only use equity indices for the canonical trading calendar.
    # Macro tickers (futures/FX) trade on different days (weekends/holidays).
    # Using them in the union injects spurious zero-return days into equities.
    equity_indices = [df.index for df in stock_data.values()]
    union_index = equity_indices[0]
    for idx in equity_indices[1:]:
        union_index = union_index.union(idx)
    
    union_index = union_index.sort_values()
    print(f"Union date range: {union_index[0]} to {union_index[-1]} "
          f"({len(union_index)} trading days)")
    
    # 5. PROCESS-FIRST, REINDEX-LAST
    # All indicators and normalization were computed on valid data in process_ticker_data().
    # Now we reindex to the union and zero-fill as the FINAL step.
    # Note: Cross-sectional demeaning and re-normalization are handled inside the
    # training loop (data_loader.py) to prevent data leakage.
    
    print("\nReindexing all data to union index and zero-filling...")
    
    # Save macro data (reindexed + zero-filled)
    for name, df in macro_data.items():
        aligned_df = df.reindex(union_index).fillna(0)
        aligned_df.to_csv(f"{PROCESSED_DIR}/macro_{name}.csv")
    
    # Save stock data (reindexed + zero-filled)
    valid_stocks = []
    for ticker, df in stock_data.items():
        aligned_df = df.reindex(union_index).fillna(0)
        aligned_df.to_csv(f"{PROCESSED_DIR}/stock_{ticker}.csv")
        valid_stocks.append(ticker)
    
    # Save the ordered superset ticker list
    pd.Series(valid_stocks).to_csv(
        f"{PROCESSED_DIR}/superset_tickers.csv", index=False, header=False
    )
    
    # Save union index for reference
    pd.Series(union_index).to_csv(
        f"{PROCESSED_DIR}/common_index.txt", index=False, header=False
    )
    
    print(f"\nIngestion Complete.")
    print(f"  Processed {len(valid_stocks)} stocks and {len(macro_data)} macro nodes.")
    print(f"  All data aligned to {len(union_index)} trading days (union index).")
    print(f"  Data saved to {PROCESSED_DIR}")
    print(f"  Superset tickers saved to {PROCESSED_DIR}/superset_tickers.csv")


if __name__ == "__main__":
    main()