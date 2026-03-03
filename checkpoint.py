"""
Checkpoint Manager for DGRCL Live Trading

Saves and loads everything needed for inference as a single .pt bundle:
    - model.state_dict()        — trained weights
    - model_config              — architecture hyperparameters
    - conformal_gate state      — q_hat threshold + alpha
    - covariance_estimator      — Sigma matrix
    - training metadata         — fold, dates, feature config, tickers, sectors

Usage:
    # After training:
    from checkpoint import CheckpointManager
    mgr = CheckpointManager("./checkpoints/latest.pt")
    mgr.save(model, constructor, metadata)

    # For live inference:
    mgr = CheckpointManager("./checkpoints/latest.pt")
    bundle = mgr.load()
    model = bundle.build_model(device)
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from macro_dgrcl import MacroDGRCL


# =============================================================================
# CHECKPOINT BUNDLE (DATACLASS-STYLE)
# =============================================================================

class CheckpointBundle:
    """
    All state needed to run live inference without retraining.
    Returned by CheckpointManager.load().
    """

    def __init__(self, data: Dict):
        self._data = data

    # --- Model ---

    def build_model(self, device: torch.device) -> MacroDGRCL:
        """Reconstruct and load the trained model onto device."""
        cfg = self._data["model_config"]
        model = MacroDGRCL(
            num_stocks=cfg["num_stocks"],
            num_macros=cfg["num_macros"],
            stock_feature_dim=cfg["stock_feature_dim"],
            macro_feature_dim=cfg["macro_feature_dim"],
            hidden_dim=cfg["hidden_dim"],
            temporal_layers=cfg["temporal_layers"],
            mp_layers=cfg["mp_layers"],
            heads=cfg["heads"],
            top_k=cfg["top_k"],
            dropout=cfg["dropout"],
            head_dropout=cfg["head_dropout"],
        ).to(device)
        model.load_state_dict(self._data["model_state"])
        model.eval()
        return model

    @property
    def model_config(self) -> Dict:
        return self._data["model_config"]

    # --- Conformal gate ---

    @property
    def conformal_q_hat(self) -> Optional[float]:
        return self._data.get("conformal_q_hat")

    @property
    def conformal_alpha(self) -> float:
        return self._data.get("conformal_alpha", 0.10)

    @property
    def conformal_calibrated(self) -> bool:
        return self._data.get("conformal_calibrated", False)

    # --- Covariance ---

    @property
    def covariance_sigma(self) -> Optional[np.ndarray]:
        sigma = self._data.get("covariance_sigma")
        if sigma is None:
            return None
        return np.array(sigma, dtype=np.float32)

    # --- Correlation graph edges ---

    @property
    def corr_edge_index(self) -> Optional[torch.Tensor]:
        data = self._data.get("corr_edge_index")
        if data is None:
            return None
        return torch.tensor(data, dtype=torch.long)

    @property
    def corr_edge_weight(self) -> Optional[torch.Tensor]:
        data = self._data.get("corr_edge_weight")
        if data is None:
            return None
        return torch.tensor(data, dtype=torch.float32)

    # --- Training metadata ---

    @property
    def tickers(self) -> List[str]:
        return self._data.get("tickers", [])

    @property
    def sector_ids(self) -> Optional[np.ndarray]:
        ids = self._data.get("sector_ids")
        return np.array(ids, dtype=np.int32) if ids is not None else None

    @property
    def feature_indices(self) -> List[int]:
        return self._data.get("feature_indices", list(range(8)))

    @property
    def feature_names(self) -> List[str]:
        return self._data.get("feature_names", [
            "Close", "High", "Low", "Log_Vol", "RSI_14", "MACD", "Volatility_5", "Returns"
        ])

    @property
    def fold_index(self) -> Optional[int]:
        return self._data.get("fold_index")

    @property
    def train_end_date(self) -> Optional[str]:
        return self._data.get("train_end_date")

    @property
    def saved_at(self) -> Optional[str]:
        return self._data.get("saved_at")

    @property
    def regime(self) -> Optional[str]:
        return self._data.get("regime")

    @property
    def realized_vol(self) -> Optional[float]:
        return self._data.get("realized_vol")


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """
    Saves and loads DGRCL inference bundles.

    The bundle is stored as a single .pt file via torch.save(), which handles
    both the model state dict (tensors) and all metadata (Python objects).
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def save(
        self,
        model: MacroDGRCL,
        model_config: Dict,
        metadata: Optional[Dict] = None,
        conformal_q_hat: Optional[float] = None,
        conformal_alpha: float = 0.10,
        conformal_calibrated: bool = False,
        covariance_sigma: Optional[np.ndarray] = None,
        corr_edge_index: Optional[torch.Tensor] = None,
        corr_edge_weight: Optional[torch.Tensor] = None,
    ):
        """
        Persist a full inference bundle to disk.

        Args:
            model:               Trained MacroDGRCL (eval mode recommended)
            model_config:        Dict of constructor kwargs (num_stocks, hidden_dim, …)
            metadata:            Optional dict of extra fields (tickers, sector_ids, etc.)
            conformal_q_hat:     Calibrated conformal threshold
            conformal_alpha:     Miscoverage level (default 0.10)
            conformal_calibrated: Whether gate has been calibrated
            covariance_sigma:    [N, N] Ledoit-Wolf covariance matrix
            corr_edge_index:     [2, E] correlation graph edges
            corr_edge_weight:    [E] correlation edge weights
        """
        bundle: Dict[str, Any] = {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "conformal_q_hat": conformal_q_hat,
            "conformal_alpha": conformal_alpha,
            "conformal_calibrated": conformal_calibrated,
            "saved_at": datetime.utcnow().isoformat() + "Z",
        }

        if covariance_sigma is not None:
            bundle["covariance_sigma"] = covariance_sigma.tolist()

        if corr_edge_index is not None:
            bundle["corr_edge_index"] = corr_edge_index.cpu().tolist()

        if corr_edge_weight is not None:
            bundle["corr_edge_weight"] = corr_edge_weight.cpu().tolist()

        if metadata:
            for k, v in metadata.items():
                if isinstance(v, np.ndarray):
                    bundle[k] = v.tolist()
                elif isinstance(v, torch.Tensor):
                    bundle[k] = v.cpu().tolist()
                else:
                    bundle[k] = v

        torch.save(bundle, self.path)
        size_mb = os.path.getsize(self.path) / 1_048_576
        print(f"Checkpoint saved: {self.path}  ({size_mb:.1f} MB)")

    def load(self) -> CheckpointBundle:
        """Load and return a CheckpointBundle from disk."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.path}\n"
                "Run train.py with --save-checkpoint to create one."
            )
        data = torch.load(self.path, map_location="cpu", weights_only=False)
        bundle = CheckpointBundle(data)
        print(
            f"Checkpoint loaded: {self.path}  "
            f"(saved {bundle.saved_at}, fold {bundle.fold_index}, "
            f"{len(bundle.tickers)} tickers)"
        )
        return bundle

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def metadata_json(self) -> str:
        """Return a JSON-safe summary of the checkpoint (no tensors/weights)."""
        bundle = self.load()
        summary = {
            "saved_at": bundle.saved_at,
            "fold_index": bundle.fold_index,
            "train_end_date": bundle.train_end_date,
            "regime": bundle.regime,
            "realized_vol": bundle.realized_vol,
            "tickers": bundle.tickers,
            "feature_names": bundle.feature_names,
            "conformal_calibrated": bundle.conformal_calibrated,
            "conformal_alpha": bundle.conformal_alpha,
            "conformal_q_hat": bundle.conformal_q_hat,
            "has_covariance": bundle.covariance_sigma is not None,
            "has_corr_edges": bundle.corr_edge_index is not None,
            "model_config": bundle.model_config,
        }
        return json.dumps(summary, indent=2)
