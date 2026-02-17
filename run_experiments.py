#!/usr/bin/env python3
"""
Experiment Runner for MacroDGRCL Feature Ablation

Runs the 4 defined feature ablation experiments sequentially:
1. baseline (All features)
2. stationary_only (No non-stationary features)
3. pure_momentum (Momentum + Returns)
4. pure_volatility (Volatility + Returns)

Aggregates results and generates comparison charts.
"""

import os
import sys
import json
import subprocess
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

# Constants
EXPERIMENTS = [
    "baseline",
    "stationary_only",
    "pure_momentum",
    "pure_volatility"
]

BASE_OUTPUT_DIR = "./backtest_results"

def run_command(cmd: List[str]):
    """Run a shell command and print output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error executing command: {' '.join(cmd)}")
        sys.exit(result.returncode)

def run_experiments(fast_mode: bool = False, cpu_mode: bool = False, use_synthetic: bool = False, 
                    epochs: int = 100, end_fold: int = None):
    """Run all ablation experiments."""
    python_exec = sys.executable
    
    for exp_name in EXPERIMENTS:
        print(f"\n\n{'='*60}")
        print(f"STARTING EXPERIMENT: {exp_name}")
        print(f"{'='*60}")
        
        output_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)
        
        # Determine parameters
        current_epochs = 1 if fast_mode else epochs
        current_end_fold = 1 if fast_mode else end_fold
        
        cmd = [
            python_exec, "train.py",
            "--ablation", exp_name,
            "--output-dir", output_dir,
            "--epochs", str(current_epochs)
        ]
        
        if current_end_fold is not None:
            cmd.extend(["--end-fold", str(current_end_fold)])
            
        if not use_synthetic:
            cmd.append("--real-data")

        if fast_mode:
            print("  [FAST MODE] Running only Fold 1, 1 Epoch")
            
        if cpu_mode:
            cmd.append("--cpu")
            
        run_command(cmd)
        print(f"\nFinished experiment: {exp_name}")

def aggregate_results() -> pd.DataFrame:
    """Aggregate results from all experiment subdirectories."""
    results = []
    
    for exp_name in EXPERIMENTS:
        path = os.path.join(BASE_OUTPUT_DIR, exp_name, "fold_results.json")
        if not os.path.exists(path):
            print(f"Warning: No results found for {exp_name} at {path}")
            continue
            
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                
            # Aggregate across folds
            avg_rank_acc = np.mean([d['rank_accuracy'] for d in data])
            avg_mag_mae = np.mean([d['mag_mae'] for d in data])
            avg_val_loss = np.mean([d['val_loss'] for d in data if d['val_loss'] is not None])
            
            results.append({
                "Experiment": exp_name,
                "Rank Accuracy": avg_rank_acc,
                "Mag MAE": avg_mag_mae,
                "Val Loss": avg_val_loss,
                "Folds": len(data)
            })
        except Exception as e:
            print(f"Error reading results for {exp_name}: {e}")
            
    df = pd.DataFrame(results)
    return df

def visualize_comparison(df: pd.DataFrame, output_dir: str):
    """Generate comparison plots."""
    if df.empty:
        print("No data to visualize.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Rank Accuracy Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["Experiment"], df["Rank Accuracy"] * 100, color='skyblue', edgecolor='black')
    plt.title("Rank Accuracy by Feature Set", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.ylim(40, 60) # Zoom in on relevant range (random=50%)
    plt.grid(axis='y', alpha=0.3)
    
    # Add values on top
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_accuracy.png"), dpi=150)
    plt.close()
    
    # 2. Magnitude MAE Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df["Experiment"], df["Mag MAE"], color='salmon', edgecolor='black')
    plt.title("Magnitude MAE by Feature Set (Lower is Better)", fontsize=14)
    plt.ylabel("MAE", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                 f'{height:.4f}', ha='center', va='bottom')
                 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_mae.png"), dpi=150)
    plt.close()
    
    print(f"\nSaved comparison plots to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Run DGRCL Feature Ablation Experiments")
    parser.add_argument("--fast", action="store_true", help="Run only 1 fold per experiment for testing")
    parser.add_argument("--skip-run", action="store_true", help="Skip training, only aggregate existing results")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of real data")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs per experiment (default: 100)")
    parser.add_argument("--end-fold", type=int, default=None, help="Limit number of folds (default: All)")
    
    args = parser.parse_args()
    
    if not args.skip_run:
        run_experiments(
            fast_mode=args.fast, 
            cpu_mode=args.cpu, 
            use_synthetic=args.synthetic,
            epochs=args.epochs,
            end_fold=args.end_fold
        )
        
    print("\n\n" + "="*60)
    print("RESULTS AGGREGATION")
    print("="*60)
    
    df = aggregate_results()
    if not df.empty:
        # Save CSV
        csv_path = os.path.join(BASE_OUTPUT_DIR, "comparison_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary table to {csv_path}")
        
        # Print table
        print("\nExperiment Summary:")
        # Format for nicer printing
        print_df = df.copy()
        print_df["Rank Accuracy"] = (print_df["Rank Accuracy"] * 100).map("{:.2f}%".format)
        print_df["Mag MAE"] = print_df["Mag MAE"].map("{:.4f}".format)
        print_df["Val Loss"] = print_df["Val Loss"].map("{:.4f}".format)
        print(print_df.to_string(index=False))
        
        # Visualize
        visualize_comparison(df, BASE_OUTPUT_DIR)
    else:
        print("No results found.")

if __name__ == "__main__":
    main()
