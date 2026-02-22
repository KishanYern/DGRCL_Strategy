import pandas as pd
import yfinance as yf
import os

def fetch_sectors():
    print("Fetching sector mapping for superset...")
    
    # Load superset
    superset_path = "./data/processed/superset_tickers.csv"
    if not os.path.exists(superset_path):
        print(f"Error: {superset_path} not found.")
        return
        
    df = pd.read_csv(superset_path, header=None)
    tickers = df[0].tolist()
    
    # Fetch sectors
    records = []
    for i, ticker in enumerate(tickers):
        print(f"[{i+1}/{len(tickers)}] Fetching sector for {ticker}...")
        try:
            info = yf.Ticker(ticker).info
            sector = info.get('sector', 'Unknown')
            if sector:
                records.append({'ticker': ticker, 'sector': sector})
            else:
                records.append({'ticker': ticker, 'sector': 'Unknown'})
        except Exception as e:
            print(f"  Failed for {ticker}: {e}")
            records.append({'ticker': ticker, 'sector': 'Unknown'})
            
    # Save mapping
    out_df = pd.DataFrame(records)
    out_path = "./data/ticker_sector_map.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved sector map to {out_path}")

if __name__ == "__main__":
    fetch_sectors()
