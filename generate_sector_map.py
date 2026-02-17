import pandas as pd
import requests
from io import StringIO
import os

def generate_sector_map(output_path="./data/processed/ticker_sector_map.csv"):
    print("Scraping S&P 500 tickers and sectors from Wikipedia...")
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'}
    
    response = requests.get(url, headers=headers)
    
    # Read the first table on the page
    table = pd.read_html(StringIO(response.text))[0]
    
    # Replace dots with dashes for Yahoo Finance compatibility (e.g., BRK.B -> BRK-B)
    table['Symbol'] = table['Symbol'].str.replace('.', '-', regex=False)
    
    # Extract only the ticker and sector columns
    sector_map = table[['Symbol', 'GICS Sector']].copy()
    sector_map.columns = ['ticker', 'sector']
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    sector_map.to_csv(output_path, index=False)
    print(f"Successfully generated sector map for {len(sector_map)} tickers.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_sector_map()