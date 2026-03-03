"""
Periodic Model Refresh for DGRCL Live Trading

Re-trains the model on the most recent data window, saves a new inference
checkpoint, and optionally signals live_trader.py to reload it.

Recommended schedule: weekly (every Monday before market open).

Usage:
    # Re-ingest latest data + retrain on last 2 folds + save checkpoint:
    python retrain.py

    # Skip data re-ingestion (use cached CSV files):
    python retrain.py --skip-ingest

    # Dry run — show what would be trained, don't save:
    python retrain.py --dry-run

    # Override checkpoint output directory:
    python retrain.py --checkpoint-dir ./checkpoints/weekly

Design:
    The walk-forward backtest covers 2007→present.  For live trading we only
    need the most recent fold:
        Training window: last TRAIN_SIZE (200) snapshots
        Validation window: last VAL_SIZE (100) snapshots
    We run train.py with --start-fold pointing to the last fold, which trains
    a fresh model on the latest data and saves a checkpoint bundle.
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — must match train.py constants
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS = 100
DEFAULT_CHECKPOINT_DIR = "./checkpoints"
DEFAULT_OUTPUT_DIR = "./backtest_results/retrain_latest"
DEFAULT_PORTFOLIO_METHOD = "mvo"
DEFAULT_PORTFOLIO_GAMMA = 1.0


def run_ingest():
    """Re-download the latest market data using data_ingest.py."""
    logger.info("Re-ingesting market data (this may take several minutes)...")
    result = subprocess.run(
        [sys.executable, "data_ingest.py"],
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"data_ingest.py exited with code {result.returncode}. "
            "Check the output above for errors."
        )
    logger.info("Data ingestion complete.")


def get_last_fold_index() -> int:
    """
    Determine the index of the last walk-forward fold from the most recent
    fold_results.json, if available.  Falls back to 90 (the full 90-fold run).
    """
    candidate_paths = [
        "./backtest_results/phase1_mvo_v2/fold_results.json",
        "./backtest_results/phase1_mvo/fold_results.json",
        "./backtest_results/fold_results.json",
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            try:
                import json
                with open(path) as f:
                    data = json.load(f)
                if data:
                    last_fold = max(r.get("fold", 0) for r in data)
                    logger.info("Detected %d folds in %s", last_fold, path)
                    return last_fold
            except Exception:
                pass
    logger.warning("Could not detect fold count — defaulting to fold 90")
    return 90


def retrain(
    skip_ingest: bool = False,
    dry_run: bool = False,
    epochs: int = DEFAULT_EPOCHS,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    portfolio_method: str = DEFAULT_PORTFOLIO_METHOD,
    portfolio_gamma: float = DEFAULT_PORTFOLIO_GAMMA,
    start_fold: int = None,
):
    """
    Run data ingest (optional) + train last fold + save checkpoint.

    Args:
        skip_ingest:      Skip data_ingest.py (use cached CSVs)
        dry_run:          Print what would run without executing
        epochs:           Training epochs (default 100)
        checkpoint_dir:   Where to save the inference checkpoint
        output_dir:       Where to save training artifacts
        portfolio_method: mvo | riskparity | naive
        portfolio_gamma:  MVO risk aversion
        start_fold:       Which fold to train (None = auto-detect last fold)
    """
    started_at = datetime.utcnow().isoformat() + "Z"
    logger.info("=== DGRCL Retrain started at %s ===", started_at)

    if not skip_ingest:
        if dry_run:
            logger.info("[DRY RUN] Would run: python data_ingest.py")
        else:
            run_ingest()
    else:
        logger.info("Skipping data ingestion (--skip-ingest)")

    # Determine fold to train
    if start_fold is None:
        start_fold = get_last_fold_index()
    logger.info("Training fold %d → %d", start_fold, start_fold)

    # Build train.py command
    cmd = [
        sys.executable, "train.py",
        "--real-data",
        f"--start-fold={start_fold}",
        f"--end-fold={start_fold}",
        f"--epochs={epochs}",
        f"--output-dir={output_dir}",
        f"--portfolio-method={portfolio_method}",
        f"--portfolio-gamma={portfolio_gamma}",
        "--save-checkpoint",
        f"--checkpoint-dir={checkpoint_dir}",
    ]

    logger.info("Training command: %s", " ".join(cmd))

    if dry_run:
        logger.info("[DRY RUN] Would run: %s", " ".join(cmd))
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"train.py exited with code {result.returncode}. "
            "Check the output above for errors."
        )

    ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / 1_048_576
        logger.info("Checkpoint saved: %s (%.1f MB)", ckpt_path, size_mb)
    else:
        logger.warning("Expected checkpoint not found at %s", ckpt_path)

    # Write a retrain log entry
    log_path = os.path.join(checkpoint_dir, "retrain_log.json")
    try:
        import json
        entry = {
            "retrained_at": started_at,
            "completed_at": datetime.utcnow().isoformat() + "Z",
            "fold": start_fold,
            "epochs": epochs,
            "portfolio_method": portfolio_method,
            "checkpoint": ckpt_path,
        }
        history = []
        if os.path.exists(log_path):
            with open(log_path) as f:
                history = json.load(f)
        history.append(entry)
        history = history[-52:]  # keep 1 year of weekly retrains
        with open(log_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info("Retrain log updated: %s", log_path)
    except Exception as e:
        logger.warning("Could not write retrain log: %s", e)

    logger.info("=== Retrain complete. Live inference will use the new checkpoint. ===")
    logger.info(
        "Restart live_trader.py (or it will pick up the new checkpoint on next cycle "
        "if InferenceEngine.reload_checkpoint() is called)."
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="DGRCL periodic model refresh — retrain on latest data and save checkpoint"
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip data_ingest.py (use cached CSV files in ./data/processed/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands that would be run without actually executing them",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help=f"Directory to write inference checkpoint (default: {DEFAULT_CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for training artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--portfolio-method",
        type=str,
        default=DEFAULT_PORTFOLIO_METHOD,
        choices=["mvo", "riskparity", "naive"],
        help=f"Portfolio method baked into the checkpoint (default: {DEFAULT_PORTFOLIO_METHOD})",
    )
    parser.add_argument(
        "--portfolio-gamma",
        type=float,
        default=DEFAULT_PORTFOLIO_GAMMA,
        help=f"MVO risk aversion gamma (default: {DEFAULT_PORTFOLIO_GAMMA})",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Specific fold to train (default: auto-detect last fold from fold_results.json)",
    )

    args = parser.parse_args()

    try:
        retrain(
            skip_ingest=args.skip_ingest,
            dry_run=args.dry_run,
            epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            portfolio_method=args.portfolio_method,
            portfolio_gamma=args.portfolio_gamma,
            start_fold=args.fold,
        )
    except Exception as e:
        logger.error("Retrain failed: %s", e)
        sys.exit(1)
