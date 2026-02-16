# DGRCL Setup Guide

## 1. Environment Setup

### Create a Virtual Environment

DGRCL requires **Python 3.10+** (Python 3.11/3.12 recommended).

```bash
# Ubuntu / Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install torch-geometric numpy pandas matplotlib seaborn yfinance scikit-learn
```

> **Note for AMD GPU Users (ROCm)**:
> Use the official ROCm PyTorch wheel:
> `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0`

---

## 2. Data Ingestion

The repository includes a script to download and process S&P 500 + Macro data from **Yahoo Finance**.

```bash
# Downloads data from Jan 2021 to Present
python data_ingest.py
```

### What happens during ingestion?
1.  **Macro Nodes**: Downloads `^VIX`, `CL=F` (Oil), `^TNX` (10Y Yields), `DX-Y.NYB` (USD Index).
2.  **S&P 500**: Scrapes the component list from Wikipedia and downloads OHLCV data.
3.  **Feature Engineering**: Calculates RSI, MACD, Volatility (5-day log returns).
4.  **Normalization**: Applies 60-day Rolling Z-Score normalization.
5.  **Alignment**: Aligns all dataframes to a common intersection of dates.
6.  **Market Neutrality**: Subtracts the cross-sectional mean return from every stock at each timestep.

The processed data will be saved in `data/processed/` as CSV files.

---

## 3. Training

### Quick Start (Synthetic Data)
To verify the installation works without downloading massive datasets:

```bash
python train.py
```

### Full Training (Real Data)
To train the model on the downloaded market data:

```bash
python train.py --real-data
```

### Walk-Forward Backtest
To simulate a realistic trading strategy over time (e.g., 2021-2026):

```bash
# Run over 15 folds (approx 5 years)
# Each fold: Train 200 days, Validate 100 days, Step 50 days
python train.py --real-data --start-fold 1 --end-fold 15
```

---

## 4. Hardware Requirements

- **CPU**: 4+ cores recommended for data loading (dataloader uses 4 workers).
- **RAM**: 16GB minimum (dataset is ~2GB in memory but overhead is high during graph construction).
- **GPU**: NVIDIA (CUDA 11+) or AMD (ROCm 5.0+).  
  - VRAM: >6GB recommended.
  - Training speed: ~1-2 seconds per epoch on modern GPU.
