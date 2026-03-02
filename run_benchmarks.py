#!/usr/bin/env python3
"""
Benchmark Runner for Macro-Aware DGRCL (Phase 0 — Rec 11)

Evaluates all benchmark models on the same 90-fold walk-forward splits used
to train DGRCL, enabling apples-to-apples comparison of:
  - Rank Accuracy
  - Sector-Balanced L/S Alpha
  - Per-regime breakdown (calm/normal/crisis)

Usage:
    # Smoke test (1 fold, synthetic data):
    python run_benchmarks.py --fast --synthetic

    # Full run (real data, all 90 folds):
    python run_benchmarks.py --real-data

    # Select specific benchmarks:
    python run_benchmarks.py --real-data --benchmarks random momentum_12_1 lstm_only

    # Run only cheap baselines (no training required):
    python run_benchmarks.py --real-data --skip-trainable
"""

import os

# AMD RDNA2 GPUs (e.g. RX 6600) need this env var so ROCm uses the correct
# gfx10.3 ISA. Must be set before any PyTorch/HIP import.
if not os.environ.get('HSA_OVERRIDE_GFX_VERSION'):
    os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import gc
import sys
import json
import math
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import torch

from benchmark_models import (
    random_scores,
    prior_day_persistence,
    momentum_12_1,
    short_term_reversal,
    low_volatility,
    LSTMOnlyBenchmark,
    LightGBMRanker,
    download_ff3_factors,
    compute_ff3_attribution,
    SIMPLE_BENCHMARKS,
    TRAINABLE_BENCHMARKS,
    ALL_BENCHMARKS,
)
from train import (
    compute_pairwise_ranking_loss,
    compute_long_short_alpha,
    classify_regime,
    compute_regime_vol,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_OUTPUT_DIR = "./backtest_results/benchmarks"

# Mapping from benchmark name to tier label for reporting
BENCHMARK_TIERS = {
    'random':               'Tier 1 — Null',
    'prior_day_persistence':'Tier 1 — Null',
    'momentum_12_1':        'Tier 2 — Classical',
    'short_term_reversal':  'Tier 2 — Classical',
    'low_volatility':       'Tier 2 — Classical',
    'lstm_only':            'Tier 3 — ML',
    'lgbm_ranker':          'Tier 3 — ML',
}

BENCHMARK_DISPLAY_NAMES = {
    'random':               'Random (Chance)',
    'prior_day_persistence':'Prior-Day Persistence',
    'momentum_12_1':        'Momentum 12-1M',
    'short_term_reversal':  'Short-Term Reversal (5d)',
    'low_volatility':       'Low Volatility',
    'lstm_only':            'LSTM-Only (No Graph)',
    'lgbm_ranker':          'LightGBM LambdaRank',
    'dgrcl':                'DGRCL v1.6 (Full Model)',
}


# =============================================================================
# PER-SNAPSHOT SCORING
# =============================================================================

def score_snapshot(
    benchmark_name: str,
    snapshot: Tuple,
    trained_model=None,
) -> torch.Tensor:
    """
    Compute benchmark scores for a single snapshot.

    Args:
        benchmark_name: Name key from ALL_BENCHMARKS
        snapshot: (stock_window, macro_window, returns, active_mask) 4-tuple
        trained_model: Pre-trained model instance (for lstm_only, lgbm_ranker)

    Returns:
        [N_s] score tensor — higher = predicted outperformer
    """
    stock_window = snapshot[0]      # [N_s, T, d_s]
    active_mask = snapshot[3] if len(snapshot) == 4 else None
    n_stocks = stock_window.size(0)

    if benchmark_name == 'random':
        return random_scores(n_stocks, active_mask)

    elif benchmark_name == 'prior_day_persistence':
        return prior_day_persistence(stock_window, active_mask)

    elif benchmark_name == 'momentum_12_1':
        return momentum_12_1(stock_window, active_mask)

    elif benchmark_name == 'short_term_reversal':
        return short_term_reversal(stock_window, active_mask)

    elif benchmark_name == 'low_volatility':
        return low_volatility(stock_window, active_mask)

    elif benchmark_name == 'lstm_only':
        assert trained_model is not None, "lstm_only requires a trained LSTMOnlyBenchmark"
        return trained_model.score(stock_window, active_mask)

    elif benchmark_name == 'lgbm_ranker':
        assert trained_model is not None, "lgbm_ranker requires a trained LightGBMRanker"
        return trained_model.score(stock_window, active_mask)

    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")


# =============================================================================
# L/S ALPHA FOR SIMPLE BENCHMARKS
# =============================================================================

def compute_benchmark_ls_alpha(
    val_snapshots: List[Tuple],
    benchmark_name: str,
    sector_ids: Optional[torch.Tensor],
    trained_model=None,
    top_n_pct: float = 0.20,
) -> Dict[str, float]:
    """
    Compute sector-balanced L/S alpha for a benchmark model.

    Mirrors train.compute_long_short_alpha() but for non-nn.Module benchmarks.

    Args:
        val_snapshots: Validation fold 4-tuples
        benchmark_name: Benchmark name key
        sector_ids: [N_s] long tensor
        trained_model: Pre-trained model (for trainable benchmarks)
        top_n_pct: Fraction of stocks per sector in L and S books

    Returns:
        Dict with total_ls_alpha, mean_ls_alpha, n_snapshots, ls_returns_list
    """
    total_spread = 0.0
    n_valid = 0
    ls_returns_list = []

    for snap in val_snapshots:
        returns = snap[2]       # [N_s]
        active_mask = snap[3] if len(snap) == 4 else None

        scores = score_snapshot(benchmark_name, snap, trained_model)

        # Determine eligible stocks
        if active_mask is not None:
            eligible = active_mask.bool()
        else:
            eligible = torch.ones(returns.shape[0], dtype=torch.bool)

        # Sector-balanced L/S book
        if sector_ids is not None and sector_ids.numel() > 0:
            unique_sectors = sector_ids.unique()
            long_returns = []
            short_returns = []

            for sid in unique_sectors:
                in_sector = (sector_ids == sid) & eligible
                if in_sector.sum() < 4:
                    continue

                sector_scores = scores[in_sector]
                sector_rets = returns[in_sector]

                # Replace -inf with very small value for argsort stability
                finite_mask = torch.isfinite(sector_scores)
                if finite_mask.sum() < 2:
                    continue

                k = int(max(1, math.floor(top_n_pct * in_sector.sum().item())))
                sorted_idx = sector_scores.argsort(descending=True)
                long_returns.append(sector_rets[sorted_idx[:k]].mean().item())
                short_returns.append(sector_rets[sorted_idx[-k:]].mean().item())

            if long_returns and short_returns:
                spread = float(np.mean(long_returns)) - float(np.mean(short_returns))
                total_spread += spread
                ls_returns_list.append(spread)
                n_valid += 1
        else:
            # Global fallback
            eligible_idx = eligible.nonzero(as_tuple=True)[0]
            if eligible_idx.numel() < 4:
                continue
            elig_scores = scores[eligible_idx]
            elig_rets = returns[eligible_idx]
            k = int(max(1, math.floor(top_n_pct * eligible_idx.numel())))
            sorted_idx = elig_scores.argsort(descending=True)
            spread = (elig_rets[sorted_idx[:k]].mean() - elig_rets[sorted_idx[-k:]].mean()).item()
            total_spread += spread
            ls_returns_list.append(spread)
            n_valid += 1

    return {
        'total_ls_alpha': total_spread,
        'mean_ls_alpha': total_spread / max(n_valid, 1),
        'n_snapshots': n_valid,
        'ls_returns_list': ls_returns_list,
    }


# =============================================================================
# FOLD EVALUATION (SINGLE BENCHMARK)
# =============================================================================

def evaluate_benchmark_on_fold(
    benchmark_name: str,
    val_snapshots: List[Tuple],
    sector_mask: Optional[torch.Tensor],
    sector_ids: Optional[torch.Tensor],
    trained_model=None,
) -> Dict:
    """
    Evaluate one benchmark on one fold's validation snapshots.

    Returns per-fold metrics: rank_accuracy, mean_ls_alpha, ls_returns_list
    """
    total_rank_acc = 0.0
    n_valid_snaps = 0

    for snap in val_snapshots:
        returns = snap[2]
        active_mask = snap[3] if len(snap) == 4 else None

        scores = score_snapshot(benchmark_name, snap, trained_model)

        # Compute pairwise ranking accuracy using same function as DGRCL
        _, rank_acc = compute_pairwise_ranking_loss(
            scores=scores,
            returns=returns,
            sector_mask=sector_mask,
            active_mask=active_mask,
        )
        total_rank_acc += rank_acc
        n_valid_snaps += 1

    mean_rank_acc = total_rank_acc / max(n_valid_snaps, 1)

    ls_result = compute_benchmark_ls_alpha(
        val_snapshots, benchmark_name, sector_ids,
        trained_model=trained_model
    )

    return {
        'rank_accuracy': mean_rank_acc,
        'mean_ls_alpha': ls_result['mean_ls_alpha'],
        'total_ls_alpha': ls_result['total_ls_alpha'],
        'n_snapshots': ls_result['n_snapshots'],
        'ls_returns_list': ls_result['ls_returns_list'],
    }


# =============================================================================
# MAIN BENCHMARK LOOP
# =============================================================================

def run_all_benchmarks(
    folds: List,
    sector_mask: Optional[torch.Tensor],
    sector_ids: Optional[torch.Tensor],
    tickers: List[str],
    dates,
    device: torch.device,
    benchmarks_to_run: List[str],
    start_fold: int = 1,
    end_fold: Optional[int] = None,
    num_stocks: int = 150,
    output_dir: str = BASE_OUTPUT_DIR,
    dgrcl_results_path: Optional[str] = None,
    ff3_factors: Optional["pd.DataFrame"] = None,
) -> Dict[str, List[Dict]]:
    """
    Run all requested benchmarks across all walk-forward folds.

    Args:
        folds: List of (train_snapshots, val_snapshots, WalkForwardFold) 3-tuples
        sector_mask: [N_s, N_s] bool
        sector_ids: [N_s] long
        tickers: List of ticker strings
        dates: pd.DatetimeIndex
        device: Torch device (used for LSTM training)
        benchmarks_to_run: Subset of ALL_BENCHMARKS to evaluate
        start_fold: First fold (1-based)
        end_fold: Last fold inclusive (None = all)
        num_stocks: Universe size (for LSTM init)
        output_dir: Directory for per-benchmark JSON results
        dgrcl_results_path: Path to DGRCL fold_results.json for comparison
        ff3_factors: Pre-downloaded FF3 factor DataFrame (or None)

    Returns:
        Dict mapping benchmark_name -> list of per-fold result dicts
    """
    os.makedirs(output_dir, exist_ok=True)

    # Map fold dates for FF3 attribution
    fold_dates_map: Dict[int, List] = {}  # fold_idx -> list of val snapshot dates

    all_results: Dict[str, List[Dict]] = {name: [] for name in benchmarks_to_run}

    for fold_idx, fold_tuple in enumerate(folds):
        current_fold_num = fold_idx + 1

        if current_fold_num < start_fold:
            continue
        if end_fold is not None and current_fold_num > end_fold:
            break

        train_snapshots = fold_tuple[0]
        val_snapshots = fold_tuple[1]
        fold_meta = fold_tuple[2] if len(fold_tuple) == 3 else None

        if not val_snapshots:
            print(f"  Fold {current_fold_num}: No validation snapshots, skipping.")
            continue

        # Compute regime for reporting
        realized_vol = compute_regime_vol(train_snapshots, lookback=20)
        regime = classify_regime(realized_vol)

        print(f"\n{'='*60}")
        print(f"FOLD {current_fold_num}/{len(folds)}  |  Regime: {regime.upper()}")
        print(f"{'='*60}")
        print(f"  Val snapshots: {len(val_snapshots)}")

        # Build approximate fold date range for FF3 date alignment
        if dates is not None and fold_meta is not None:
            val_start_t = fold_meta.val_start
            val_end_t = fold_meta.val_end
            date_indices = range(min(val_start_t, len(dates)-1),
                                 min(val_end_t, len(dates)))
            fold_val_dates = [dates.iloc[i] for i in date_indices]
        else:
            fold_val_dates = []

        for benchmark_name in benchmarks_to_run:
            print(f"  [{benchmark_name}] ", end='', flush=True)

            trained_model = None

            # --- Train per-fold models ---
            if benchmark_name == 'lstm_only':
                stock_dim = train_snapshots[0][0].size(-1) if train_snapshots else 8
                lstm_bench = LSTMOnlyBenchmark(
                    num_stocks=num_stocks,
                    stock_feature_dim=stock_dim,
                    hidden_dim=64,
                    dropout=0.5,
                    device=device,
                )
                train_result = lstm_bench.train_fold(
                    train_snapshots,
                    sector_mask=sector_mask,
                    sector_ids=sector_ids,
                    num_epochs=100,
                    patience=10,
                )
                trained_model = lstm_bench
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                print(f"trained ({train_result['epochs_trained']} epochs) ", end='', flush=True)

            elif benchmark_name == 'lgbm_ranker':
                stock_dim = train_snapshots[0][0].size(-1) if train_snapshots else 8
                lgbm_bench = LightGBMRanker(feature_dim=stock_dim)
                lgbm_bench.train_fold(train_snapshots)
                trained_model = lgbm_bench
                print(f"trained ", end='', flush=True)

            # --- Evaluate on validation snapshots ---
            fold_metrics = evaluate_benchmark_on_fold(
                benchmark_name=benchmark_name,
                val_snapshots=val_snapshots,
                sector_mask=sector_mask,
                sector_ids=sector_ids,
                trained_model=trained_model,
            )

            fold_result = {
                'fold': current_fold_num,
                'regime': regime,
                'realized_vol': realized_vol,
                'rank_accuracy': fold_metrics['rank_accuracy'],
                'mean_ls_alpha': fold_metrics['mean_ls_alpha'],
                'total_ls_alpha': fold_metrics['total_ls_alpha'],
                'n_snapshots': fold_metrics['n_snapshots'],
                # Store L/S return series for FF3 attribution
                'ls_returns_list': fold_metrics['ls_returns_list'],
                'fold_val_dates': [str(d) for d in fold_val_dates],
            }

            all_results[benchmark_name].append(fold_result)

            print(
                f"RankAcc={fold_result['rank_accuracy']*100:.1f}% | "
                f"LS={fold_result['mean_ls_alpha']:+.4f} (z)"
            )

        # Clean up trainable models to free memory
        gc.collect()

    # --- Save per-benchmark JSON results ---
    for benchmark_name, results in all_results.items():
        json_path = os.path.join(output_dir, f"{benchmark_name}_fold_results.json")
        # Remove ls_returns_list from JSON to keep it compact (stored separately)
        compact = []
        for r in results:
            row = {k: v for k, v in r.items() if k != 'ls_returns_list' and k != 'fold_val_dates'}
            compact.append(row)
        with open(json_path, 'w') as f:
            json.dump(compact, f, indent=2)
        print(f"\nSaved: {json_path}")

    # --- Fama-French 3-Factor Attribution ---
    if ff3_factors is not None:
        print("\n\n=== Fama-French 3-Factor Attribution ===")
        print("  NOTE: L/S returns are z-score-normalized, not raw returns.")
        print("  Alpha/beta values are in z-score units — interpret directionally, not as %/yr.")
        ff3_results = {}

        # Combine all fold L/S returns for DGRCL (from existing results file)
        if dgrcl_results_path and os.path.exists(dgrcl_results_path):
            with open(dgrcl_results_path) as f:
                dgrcl_fold_results = json.load(f)
        else:
            dgrcl_fold_results = []

        for benchmark_name in benchmarks_to_run:
            # Gather L/S return series across all folds
            all_ls_rets = []
            all_ls_dates = []

            for fold_result in all_results[benchmark_name]:
                ls_rets = fold_result.get('ls_returns_list', [])
                date_strs = fold_result.get('fold_val_dates', [])
                # Truncate dates to match number of L/S returns available
                paired = list(zip(ls_rets, date_strs[:len(ls_rets)]))
                for ret, date_str in paired:
                    all_ls_rets.append(ret)
                    all_ls_dates.append(pd.Timestamp(date_str))

            if all_ls_rets:
                attr = compute_ff3_attribution(all_ls_rets, all_ls_dates, ff3_factors)
                ff3_results[benchmark_name] = attr
                print(f"  {BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name):35s} "
                      f"alpha={attr['alpha']:.4f}  "
                      f"t={attr['alpha_t_stat']:.2f}  "
                      f"R²={attr['r_squared']:.3f}  "
                      f"n={attr['n_observations']}")

        ff3_path = os.path.join(output_dir, "ff3_attribution.json")
        with open(ff3_path, 'w') as f:
            json.dump(ff3_results, f, indent=2)
        print(f"\nSaved FF3 attribution: {ff3_path}")

    return all_results


# =============================================================================
# AGGREGATION & REPORTING
# =============================================================================

def aggregate_and_report(
    all_results: Dict[str, List[Dict]],
    dgrcl_results_path: Optional[str],
    output_dir: str,
) -> pd.DataFrame:
    """
    Aggregate per-fold metrics and produce comparison summary.

    Loads DGRCL fold_results.json for direct comparison, then generates
    both a CSV summary table and a multi-panel visualization.

    Args:
        all_results: Dict from run_all_benchmarks()
        dgrcl_results_path: Path to DGRCL's fold_results.json
        output_dir: Output directory for CSV and plots

    Returns:
        Summary DataFrame with one row per benchmark
    """
    summary_rows = []

    # Load DGRCL results for comparison
    dgrcl_data = []
    if dgrcl_results_path and os.path.exists(dgrcl_results_path):
        with open(dgrcl_results_path) as f:
            dgrcl_data = json.load(f)

    if dgrcl_data:
        dgrcl_rank_acc = np.mean([d.get('rank_accuracy', 0) for d in dgrcl_data])
        # DGRCL fold_results.json uses 'ls_alpha_mean', not 'mean_ls_alpha'
        dgrcl_ls = np.mean([
            d.get('ls_alpha_mean', d.get('mean_ls_alpha', d.get('val_ls_alpha', 0)))
            for d in dgrcl_data
        ])
        regime_breakdown = _compute_regime_breakdown(dgrcl_data)
        summary_rows.append({
            'benchmark': 'dgrcl',
            'display_name': BENCHMARK_DISPLAY_NAMES['dgrcl'],
            'tier': 'DGRCL (Full)',
            'rank_accuracy': dgrcl_rank_acc,
            'mean_ls_alpha': dgrcl_ls,
            'n_folds': len(dgrcl_data),
            **{f'rank_acc_{r}': regime_breakdown.get(r, {}).get('rank_accuracy', float('nan'))
               for r in ['calm', 'normal', 'crisis']},
            **{f'ls_{r}': regime_breakdown.get(r, {}).get('mean_ls_alpha', float('nan'))
               for r in ['calm', 'normal', 'crisis']},
        })

    # Aggregate benchmark results
    for benchmark_name, fold_results in all_results.items():
        if not fold_results:
            continue

        rank_accs = [r['rank_accuracy'] for r in fold_results]
        ls_alphas = [r['mean_ls_alpha'] for r in fold_results]
        regime_breakdown = _compute_regime_breakdown(fold_results)

        summary_rows.append({
            'benchmark': benchmark_name,
            'display_name': BENCHMARK_DISPLAY_NAMES.get(benchmark_name, benchmark_name),
            'tier': BENCHMARK_TIERS.get(benchmark_name, 'Unknown'),
            'rank_accuracy': np.mean(rank_accs),
            'mean_ls_alpha': np.mean(ls_alphas),
            'n_folds': len(fold_results),
            **{f'rank_acc_{r}': regime_breakdown.get(r, {}).get('rank_accuracy', float('nan'))
               for r in ['calm', 'normal', 'crisis']},
            **{f'ls_{r}': regime_breakdown.get(r, {}).get('mean_ls_alpha', float('nan'))
               for r in ['calm', 'normal', 'crisis']},
        })

    df = pd.DataFrame(summary_rows)

    # Save CSV
    csv_path = os.path.join(output_dir, "benchmark_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison CSV: {csv_path}")

    # Print formatted table
    _print_comparison_table(df)

    # Generate visualization
    _generate_comparison_plots(df, all_results, dgrcl_data, output_dir)

    return df


def _compute_regime_breakdown(fold_results: List[Dict]) -> Dict[str, Dict]:
    """Compute mean metrics grouped by regime label.

    Handles both benchmark output format (mean_ls_alpha) and DGRCL's
    existing fold_results.json format (ls_alpha / val_ls_alpha).
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for r in fold_results:
        grouped[r.get('regime', 'normal')].append(r)

    breakdown = {}
    for regime, rows in grouped.items():
        # Support both key naming conventions
        def _rank_acc(r):
            return r.get('rank_accuracy', r.get('val_rank_accuracy', 0.0))

        def _ls(r):
            return r.get('mean_ls_alpha', r.get('ls_alpha_mean', r.get('ls_alpha', r.get('val_ls_alpha', 0.0))))

        breakdown[regime] = {
            'rank_accuracy': np.mean([_rank_acc(r) for r in rows]),
            'mean_ls_alpha': np.mean([_ls(r) for r in rows]),
            'n_folds': len(rows),
        }
    return breakdown


def _print_comparison_table(df: pd.DataFrame) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON SUMMARY")
    print("="*80)

    cols = ['display_name', 'tier', 'rank_accuracy', 'mean_ls_alpha', 'n_folds']
    display = df[cols].copy()
    display['rank_accuracy'] = (display['rank_accuracy'] * 100).map("{:.2f}%".format)
    # L/S alpha is in z-score-normalized return units (not raw percentage)
    display['mean_ls_alpha'] = display['mean_ls_alpha'].map("{:+.4f}".format)
    display.columns = ['Model', 'Tier', 'Rank Accuracy', 'L/S Spread (z)', 'Folds']

    # Sort: DGRCL first, then by rank accuracy descending
    dgrcl_row = display[display['Model'] == BENCHMARK_DISPLAY_NAMES['dgrcl']]
    others = display[display['Model'] != BENCHMARK_DISPLAY_NAMES['dgrcl']].sort_values(
        'Rank Accuracy', ascending=False
    )
    display = pd.concat([dgrcl_row, others])

    print(display.to_string(index=False))
    print("="*80)


def _generate_comparison_plots(
    df: pd.DataFrame,
    all_results: Dict[str, List[Dict]],
    dgrcl_data: List[Dict],
    output_dir: str,
) -> None:
    """
    Generate a multi-panel benchmark comparison plot.

    Panels:
      Row 1: Rank Accuracy comparison (bar) | L/S Alpha comparison (bar)
      Row 2: Per-regime rank accuracy       | Per-regime L/S alpha
      Row 3: Per-fold rank accuracy lines   | Per-fold L/S alpha lines
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Benchmark Comparison — DGRCL v1.6 vs Baselines', fontsize=14, fontweight='bold')

    # Define color palette
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(df), 1)))
    model_colors = dict(zip(df['benchmark'].tolist(), colors))
    # DGRCL gets a special gold color
    if 'dgrcl' in model_colors:
        model_colors['dgrcl'] = '#FFD700'

    display_names = df.set_index('benchmark')['display_name'].to_dict()
    names = [display_names.get(b, b) for b in df['benchmark']]

    # Panel 1: Overall Rank Accuracy
    ax1 = fig.add_subplot(3, 2, 1)
    bars = ax1.barh(names, df['rank_accuracy'] * 100,
                    color=[model_colors[b] for b in df['benchmark']], edgecolor='black', height=0.6)
    ax1.axvline(50, color='red', linestyle='--', linewidth=1.5, label='Chance (50%)')
    ax1.set_xlabel('Rank Accuracy (%)')
    ax1.set_title('Pairwise Rank Accuracy (All Folds Mean)', fontweight='bold')
    ax1.set_xlim(40, 65)
    ax1.grid(axis='x', alpha=0.3)
    ax1.legend(fontsize=9)
    for bar, val in zip(bars, df['rank_accuracy']):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 f'{val*100:.2f}%', va='center', fontsize=8)

    # Panel 2: Overall L/S Alpha
    ax2 = fig.add_subplot(3, 2, 2)
    ls_vals = df['mean_ls_alpha'] * 100
    bar_colors = ['#66BB6A' if v >= 0 else '#EF5350' for v in ls_vals]
    bars = ax2.barh(names, ls_vals, color=bar_colors, edgecolor='black', height=0.6)
    ax2.axvline(0, color='black', linewidth=1.0)
    ax2.set_xlabel('Mean L/S Spread (z-score units)')
    ax2.set_title('Sector-Balanced L/S Spread (Mean per Snapshot)', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, ls_vals):
        ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}%', va='center', fontsize=8)

    # Panel 3 & 4: Per-regime breakdown
    regimes = ['calm', 'normal', 'crisis']
    regime_colors = {'calm': '#64B5F6', 'normal': '#81C784', 'crisis': '#E57373'}
    x = np.arange(len(df))
    width = 0.25

    ax3 = fig.add_subplot(3, 2, 3)
    for i, regime in enumerate(regimes):
        col = f'rank_acc_{regime}'
        if col in df.columns:
            vals = df[col].fillna(0) * 100
            ax3.bar(x + i * width, vals, width, label=regime.capitalize(),
                    color=regime_colors[regime], edgecolor='black', alpha=0.85)
    ax3.axhline(50, color='red', linestyle='--', linewidth=1.0)
    ax3.set_xticks(x + width)
    ax3.set_xticklabels([n[:20] for n in names], rotation=45, ha='right', fontsize=8)
    ax3.set_ylabel('Rank Accuracy (%)')
    ax3.set_title('Rank Accuracy by Regime', fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    ax4 = fig.add_subplot(3, 2, 4)
    for i, regime in enumerate(regimes):
        col = f'ls_{regime}'
        if col in df.columns:
            vals = df[col].fillna(0) * 100
            ax4.bar(x + i * width, vals, width, label=regime.capitalize(),
                    color=regime_colors[regime], edgecolor='black', alpha=0.85)
    ax4.axhline(0, color='black', linewidth=1.0)
    ax4.set_xticks(x + width)
    ax4.set_xticklabels([n[:20] for n in names], rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Mean L/S Spread (z-score)')
    ax4.set_title('L/S Alpha by Regime', fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)

    # Panel 5 & 6: Per-fold rank accuracy and L/S alpha trend
    ax5 = fig.add_subplot(3, 2, 5)
    ax6 = fig.add_subplot(3, 2, 6)

    if dgrcl_data:
        dgrcl_folds = [d['fold'] for d in dgrcl_data]
        dgrcl_accs = [d.get('rank_accuracy', 0) for d in dgrcl_data]
        dgrcl_ls = [d.get('ls_alpha_mean', d.get('mean_ls_alpha', 0)) for d in dgrcl_data]
        ax5.plot(dgrcl_folds, [a*100 for a in dgrcl_accs], 'o-',
                 color='#FFD700', linewidth=2, label='DGRCL v1.6', markersize=4, zorder=5)
        ax6.plot(dgrcl_folds, [a*100 for a in dgrcl_ls], 'o-',
                 color='#FFD700', linewidth=2, label='DGRCL v1.6', markersize=4, zorder=5)

    for benchmark_name, fold_results in all_results.items():
        if not fold_results:
            continue
        folds = [r['fold'] for r in fold_results]
        accs = [r['rank_accuracy'] for r in fold_results]
        ls_alphas = [r['mean_ls_alpha'] for r in fold_results]
        c = model_colors.get(benchmark_name, 'grey')
        label = display_names.get(benchmark_name, benchmark_name)
        ax5.plot(folds, [a*100 for a in accs], 's--', color=c, linewidth=1,
                 label=label, markersize=3, alpha=0.75)
        ax6.plot(folds, [a*100 for a in ls_alphas], 's--', color=c, linewidth=1,
                 label=label, markersize=3, alpha=0.75)

    ax5.axhline(50, color='red', linestyle='--', linewidth=1.0, label='Chance (50%)')
    ax5.set_xlabel('Fold')
    ax5.set_ylabel('Rank Accuracy (%)')
    ax5.set_title('Per-Fold Rank Accuracy', fontweight='bold')
    ax5.legend(fontsize=7, ncol=2)
    ax5.grid(alpha=0.3)

    ax6.axhline(0, color='black', linewidth=1.0)
    ax6.set_xlabel('Fold')
    ax6.set_ylabel('Mean L/S Spread (z-score)')
    ax6.set_title('Per-Fold L/S Spread', fontweight='bold')
    ax6.legend(fontsize=7, ncol=2)
    ax6.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'benchmark_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot: {plot_path}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DGRCL Phase 0 Benchmark Suite (Rec 11)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--real-data', action='store_true',
                        help='Use real market data from ./data/processed/')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (overrides --real-data for smoke testing)')
    parser.add_argument('--fast', action='store_true',
                        help='Run only fold 1 for smoke testing')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU mode (skips CUDA)')
    parser.add_argument('--start-fold', type=int, default=1,
                        help='First fold to evaluate (1-based)')
    parser.add_argument('--end-fold', type=int, default=None,
                        help='Last fold to evaluate inclusive (default: all)')
    parser.add_argument('--benchmarks', nargs='+', default=None,
                        choices=ALL_BENCHMARKS,
                        help='Subset of benchmarks to run (default: all)')
    parser.add_argument('--skip-trainable', action='store_true',
                        help='Skip LSTM and LightGBM (run only Tier 1/2)')
    parser.add_argument('--skip-ff3', action='store_true',
                        help='Skip Fama-French attribution (no internet required)')
    parser.add_argument('--dgrcl-results', type=str,
                        default='./backtest_results/fold_results.json',
                        help='Path to DGRCL fold_results.json for comparison')
    parser.add_argument('--output-dir', type=str, default=BASE_OUTPUT_DIR,
                        help='Output directory for benchmark results')
    args = parser.parse_args()

    # Resolve benchmark list
    benchmarks_to_run = args.benchmarks or ALL_BENCHMARKS[:]
    if args.skip_trainable:
        benchmarks_to_run = [b for b in benchmarks_to_run if b in SIMPLE_BENCHMARKS]

    use_real_data = args.real_data and not args.synthetic

    print("=" * 60)
    print("DGRCL Phase 0 — Benchmark Suite (Rec 11)")
    print("=" * 60)
    print(f"Benchmarks: {benchmarks_to_run}")
    print(f"Data: {'Real' if use_real_data else 'Synthetic'}")
    print(f"Fast mode: {args.fast}")
    print()

    # Device
    device = torch.device('cpu') if args.cpu else \
             torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # --- Load Data ---
    if use_real_data:
        from data_loader import load_and_prepare_backtest
        from macro_dgrcl import HeteroGraphBuilder

        print("\n=== Loading Real Market Data ===")
        folds, dates, tickers, num_stocks, num_macros, sector_map = load_and_prepare_backtest(
            data_dir='./data/processed',
            train_size=200,
            val_size=100,
            step_size=50,
            window_size=60,
            forecast_horizon=5,
            snapshot_step=1,
            macro_lag=5,
            device=torch.device('cpu'),
            max_stocks=150,
        )

        graph_builder = HeteroGraphBuilder(sector_mapping=sector_map, tickers=tickers)
        sector_mask = graph_builder.sector_mask

        if sector_map and tickers:
            unique_sectors = sorted(set(sector_map.values()))
            sector_to_id = {s: i for i, s in enumerate(unique_sectors)}
            sector_ids = torch.tensor(
                [sector_to_id.get(sector_map.get(t, 'Unknown'), -1) for t in tickers],
                dtype=torch.long
            )
        else:
            sector_ids = None

        print(f"Ready: {len(folds)} folds, {num_stocks} stocks")

    else:
        # Synthetic data for smoke testing
        print("\n=== Generating Synthetic Data ===")
        from data_loader import WalkForwardSplitter, create_backtest_folds

        NUM_STOCKS = 50
        NUM_MACROS = 4
        T_TOTAL = 600

        torch.manual_seed(42)
        stock_data = torch.randn(NUM_STOCKS, T_TOTAL, 8)
        macro_data = torch.randn(NUM_MACROS, T_TOTAL, 4)
        returns_data = torch.randn(NUM_STOCKS, T_TOTAL) * 0.02
        inclusion_mask = torch.rand(NUM_STOCKS, T_TOTAL) > 0.2

        splitter = WalkForwardSplitter(train_size=200, val_size=100, step_size=50)
        folds = create_backtest_folds(
            stock_data, macro_data, returns_data, inclusion_mask, splitter,
            window_size=60, forecast_horizon=5, step_size=1
        )

        tickers = [f'STOCK_{i:03d}' for i in range(NUM_STOCKS)]
        dates = None
        num_stocks = NUM_STOCKS
        sector_mask = None
        sector_ids = None

    # Fast mode: 1 fold only
    end_fold = 1 if args.fast else args.end_fold
    start_fold = args.start_fold

    # --- Download FF3 Factors ---
    ff3_factors = None
    if not args.skip_ff3 and 'lstm_only' not in benchmarks_to_run:
        # FF3 attribution needs date-aligned L/S returns — run after full evaluation
        pass  # Downloaded inside run_all_benchmarks if needed

    if not args.skip_ff3 and use_real_data:
        print("\nDownloading Fama-French 3-Factor data...")
        try:
            ff3_factors = download_ff3_factors()
            print(f"  FF3 factors loaded: {len(ff3_factors)} trading days")
        except Exception as e:
            print(f"  WARNING: Could not download FF3 factors: {e}")
            print("  Skipping FF3 attribution. Use --skip-ff3 to suppress this warning.")
            ff3_factors = None

    # --- Run Benchmarks ---
    all_results = run_all_benchmarks(
        folds=folds,
        sector_mask=sector_mask,
        sector_ids=sector_ids,
        tickers=tickers,
        dates=dates,
        device=device,
        benchmarks_to_run=benchmarks_to_run,
        start_fold=start_fold,
        end_fold=end_fold,
        num_stocks=num_stocks,
        output_dir=args.output_dir,
        dgrcl_results_path=args.dgrcl_results,
        ff3_factors=ff3_factors,
    )

    # --- Aggregate & Report ---
    print("\n\n=== Aggregating Results ===")
    summary_df = aggregate_and_report(
        all_results=all_results,
        dgrcl_results_path=args.dgrcl_results,
        output_dir=args.output_dir,
    )

    print(f"\nBenchmark run complete. Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
