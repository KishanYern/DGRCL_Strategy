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
pip install torch-geometric numpy pandas matplotlib seaborn yfinance
```

**Optional** (for Platt Scaling calibration):
```bash
pip install scikit-learn
```

> **Note for AMD GPU Users (ROCm)**:
> Use the official ROCm PyTorch wheel:
> `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0`

---

## 2. Data Ingestion

The repository includes a script to download and process S&P 500 + Macro data from **Yahoo Finance**.

```bash
# Downloads data from Jan 2021 to Present
python3 data_ingest.py
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

### Quick Verification (Synthetic Data)
```bash
python3 train.py
```

### Full Pipeline (Recommended)
Using the unified script that runs both backtest and calibration:

```bash
# AMD GPU — sets ROCm environment automatically:
bash run_gpu_training.sh --real-data --save-calibration-data

# CPU / NVIDIA — run manually:
python3 train.py --real-data --save-calibration-data
python3 confidence_calibration.py --results-dir ./backtest_results
```

### Partial Backtest (Specific Folds)
```bash
bash run_gpu_training.sh --real-data --start-fold 1 --end-fold 10
```

### CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--real-data` | — | Use real market data from `./data/processed/` |
| `--save-calibration-data` | — | Save val logits/labels for confidence calibration |
| `--start-fold` | 1 | First fold to run (1-based) |
| `--end-fold` | 90 | Last fold to run (inclusive) |
| `--mag-weight` | 0.05 | Base λ for magnitude loss (adaptive scaling still applies) |
| `--epochs` | 100 | Max epochs per fold |
| `--output-dir` | `./backtest_results` | Results directory |
| `--cpu` | — | Force CPU training |
| `--ablation` | `baseline` | Feature ablation variant |

---

## 4. Output Files

After a full run, `backtest_results/` will contain:

| File | Description |
|------|-------------|
| `fold_results.json` | Per-fold metrics: rank accuracy, val loss, regime, λ, L/S alpha |
| `backtest_summary.png` | Multi-panel summary plot |
| `training_curve_fold_X.png` | Loss curves per fold |
| `attention_heatmap_fold_90.png` | Final fold attention weights |
| `calibration_logits.json` | Raw val logits (for post-processing) |
| `calibration_labels.json` | Binary val labels |
| `temperature_calibration.json` | Fitted T parameter + ECE before/after |
| `conformal_predictor.json` | q_hat threshold + α |
| `reliability_diagram_before.png` | Calibration plot (raw) |
| `reliability_diagram_after_temp.png` | Calibration plot (post-temperature) |

---

## 5. Hardware Requirements

- **CPU**: 4+ cores recommended for data loading.
- **RAM**: 16GB minimum.
- **GPU**: NVIDIA (CUDA 11+) or AMD (ROCm 5.0+).
  - VRAM: >6GB recommended.
  - Training speed: ~1-2 seconds per epoch on modern GPU.

### AMD GPU Specifics (RX 6600 / Navi 23)
The `run_gpu_training.sh` script sets the required environment variables:
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```
Always use `bash run_gpu_training.sh` instead of `python3 train.py` directly on AMD hardware.
