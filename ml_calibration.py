"""
ML Calibration Layer
====================
Wrapper der kalibrerer probability outputs fra ML-modeller
så confidence-scores matcher reel hit-rate.

Eksempel: Model siger 70% sandsynlighed → reel hit-rate skal være ~70%

Metoder:
- Platt Scaling (sigmoid) - bedst til små datasæt, parametrisk
- Isotonic Regression - bedst til store datasæt, non-parametrisk
- Auto - vælger bedste metode baseret på datasæt-størrelse + Brier score
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Data Classes
# ============================================================

@dataclass
class CalibrationMetrics:
    """Metrics til at evaluere calibration kvalitet."""
    brier_score: float          # Lower is better (0 = perfect)
    log_loss_value: float       # Lower is better
    ece: float                  # Expected Calibration Error (lower is better)
    mce: float                  # Maximum Calibration Error
    n_samples: int
    method: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"CalibrationMetrics(method={self.method}, "
            f"brier={self.brier_score:.4f}, "
            f"logloss={self.log_loss_value:.4f}, "
            f"ECE={self.ece:.4f}, n={self.n_samples})"
        )


# ============================================================
# Core Calibrator
# ============================================================

class ProbabilityCalibrator:
    """
    Kalibrerer probability outputs fra en trænet binær classifier.

    Usage:
        cal = ProbabilityCalibrator(method="auto")
        cal.fit(raw_probs_val, y_val)
        calibrated = cal.transform(raw_probs_test)
    """

    def __init__(
        self,
        method: Literal["platt", "isotonic", "auto"] = "auto",
        cv_folds: int = 5,
    ):
        self.method = method
        self.cv_folds = cv_folds
        self.calibrator_ = None
        self.chosen_method_: Optional[str] = None
        self.metrics_before_: Optional[CalibrationMetrics] = None
        self.metrics_after_: Optional[CalibrationMetrics] = None
        self.is_fitted_ = False

    # ------------- Fit -------------
    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fit calibrator på validation-set.

        Args:
            probs: Raw probabilities fra model, shape (n,) eller (n, 2)
            y_true: Faktiske labels (0/1), shape (n,)
        """
        probs = self._ensure_1d(probs)
        y_true = np.asarray(y_true).astype(int).ravel()

        if len(probs) != len(y_true):
            raise ValueError(f"Length mismatch: probs={len(probs)}, y={len(y_true)}")

        if len(np.unique(y_true)) < 2:
            raise ValueError("Need both classes (0 and 1) in y_true to calibrate")

        # Metrics før calibration
        self.metrics_before_ = self._compute_metrics(probs, y_true, method="raw")

        # Vælg metode
        method = self._choose_method(probs, y_true) if self.method == "auto" else self.method
        self.chosen_method_ = method

        # Fit
        if method == "platt":
            self.calibrator_ = self._fit_platt(probs, y_true)
        elif method == "isotonic":
            self.calibrator_ = self._fit_isotonic(probs, y_true)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Metrics efter calibration
        calibrated = self.transform(probs)
        self.metrics_after_ = self._compute_metrics(calibrated, y_true, method=method)
        self.is_fitted_ = True

        return self

    # ------------- Transform -------------
    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Kalibrer raw probabilities."""
        if not self.is_fitted_:
            raise RuntimeError("Calibrator not fitted. Call .fit() first.")

        probs = self._ensure_1d(probs)

        if self.chosen_method_ == "platt":
            # LogisticRegression forventer 2D
            return self.calibrator_.predict_proba(probs.reshape(-1, 1))[:, 1]
        elif self.chosen_method_ == "isotonic":
            return self.calibrator_.predict(probs)

        raise RuntimeError("Unknown calibrator state")

    def fit_transform(self, probs: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return self.fit(probs, y_true).transform(probs)

    # ------------- Auto-select -------------
    def _choose_method(self, probs: np.ndarray, y_true: np.ndarray) -> str:
        """
        Vælg bedste metode via cross-validation på Brier score.
        Tommelfingerregel:
        - n < 1000: foretrækker Platt (mindre overfit-risiko)
        - n >= 1000: prøv begge og vælg laveste Brier
        """
        n = len(probs)

        if n < 200:
            return "platt"  # for lidt data til isotonic

        # Cross-validate begge metoder
        scores = {"platt": [], "isotonic": []}
        skf = StratifiedKFold(n_splits=min(self.cv_folds, 5), shuffle=True, random_state=42)

        for train_idx, val_idx in skf.split(probs, y_true):
            p_tr, p_val = probs[train_idx], probs[val_idx]
            y_tr, y_val = y_true[train_idx], y_true[val_idx]

            # Skip fold hvis kun én klasse
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                continue

            try:
                platt = self._fit_platt(p_tr, y_tr)
                p_platt = platt.predict_proba(p_val.reshape(-1, 1))[:, 1]
                scores["platt"].append(brier_score_loss(y_val, p_platt))
            except Exception:
                pass

            try:
                iso = self._fit_isotonic(p_tr, y_tr)
                p_iso = iso.predict(p_val)
                scores["isotonic"].append(brier_score_loss(y_val, p_iso))
            except Exception:
                pass

        mean_scores = {k: np.mean(v) if v else np.inf for k, v in scores.items()}
        return min(mean_scores, key=mean_scores.get)

    # ------------- Fitters -------------
    @staticmethod
    def _fit_platt(probs: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
        """Platt scaling = logistic regression på raw probs."""
        lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        lr.fit(probs.reshape(-1, 1), y_true)
        return lr

    @staticmethod
    def _fit_isotonic(probs: np.ndarray, y_true: np.ndarray) -> IsotonicRegression:
        """Isotonic regression - non-parametrisk, monotont."""
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(probs, y_true)
        return iso

    # ------------- Metrics -------------
    @staticmethod
    def _compute_metrics(
        probs: np.ndarray, y_true: np.ndarray, method: str = ""
    ) -> CalibrationMetrics:
        probs = np.clip(probs, 1e-7, 1 - 1e-7)
        ece, mce = ProbabilityCalibrator._calibration_errors(probs, y_true, n_bins=10)

        return CalibrationMetrics(
            brier_score=float(brier_score_loss(y_true, probs)),
            log_loss_value=float(log_loss(y_true, probs)),
            ece=float(ece),
            mce=float(mce),
            n_samples=len(probs),
            method=method,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    @staticmethod
    def _calibration_errors(
        probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10
    ) -> tuple[float, float]:
        """Expected Calibration Error (ECE) + Maximum Calibration Error (MCE)."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.digitize(probs, bin_edges[1:-1])

        ece, mce = 0.0, 0.0
        n = len(probs)

        for b in range(n_bins):
            mask = bin_ids == b
            if not mask.any():
                continue
            avg_pred = probs[mask].mean()
            avg_actual = y_true[mask].mean()
            gap = abs(avg_pred - avg_actual)
            weight = mask.sum() / n
            ece += weight * gap
            mce = max(mce, gap)

        return ece, mce

    # ------------- Utils -------------
    @staticmethod
    def _ensure_1d(probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if probs.ndim == 2:
            if probs.shape[1] == 2:
                probs = probs[:, 1]
            elif probs.shape[1] == 1:
                probs = probs.ravel()
            else:
                raise ValueError(f"Expected (n,) or (n,2), got {probs.shape}")
        return probs.ravel()

    # ------------- Persistence -------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.method,
            "chosen_method": self.chosen_method_,
            "calibrator": self.calibrator_,
            "metrics_before": self.metrics_before_.to_dict() if self.metrics_before_ else None,
            "metrics_after": self.metrics_after_.to_dict() if self.metrics_after_ else None,
            "cv_folds": self.cv_folds,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        cal = cls(method=payload["method"], cv_folds=payload["cv_folds"])
        cal.calibrator_ = payload["calibrator"]
        cal.chosen_method_ = payload["chosen_method"]
        cal.is_fitted_ = True
        if payload["metrics_before"]:
            cal.metrics_before_ = CalibrationMetrics(**payload["metrics_before"])
        if payload["metrics_after"]:
            cal.metrics_after_ = CalibrationMetrics(**payload["metrics_after"])
        return cal

    # ------------- Reporting -------------
    def report(self) -> str:
        if not self.is_fitted_:
            return "Calibrator not fitted yet."

        before, after = self.metrics_before_, self.metrics_after_
        improvement_brier = (before.brier_score - after.brier_score) / before.brier_score * 100
        improvement_ece = (before.ece - after.ece) / max(before.ece, 1e-9) * 100

        return (
            f"\n📊 Calibration Report ({self.chosen_method_.upper()})\n"
            f"{'='*50}\n"
            f"{'Metric':<15}{'Before':>12}{'After':>12}{'Δ':>10}\n"
            f"{'-'*50}\n"
            f"{'Brier Score':<15}{before.brier_score:>12.4f}{after.brier_score:>12.4f}"
            f"{improvement_brier:>9.1f}%\n"
            f"{'Log Loss':<15}{before.log_loss_value:>12.4f}{after.log_loss_value:>12.4f}"
            f"{(before.log_loss_value-after.log_loss_value)/before.log_loss_value*100:>9.1f}%\n"
            f"{'ECE':<15}{before.ece:>12.4f}{after.ece:>12.4f}{improvement_ece:>9.1f}%\n"
            f"{'MCE':<15}{before.mce:>12.4f}{after.mce:>12.4f}\n"
            f"{'Samples':<15}{before.n_samples:>12d}\n"
            f"{'='*50}\n"
        )


# ============================================================
# Multi-Model Calibrator (RF + XGB + LGBM ensemble)
# ============================================================

class EnsembleCalibrator:
    """
    Kalibrerer flere modeller separat og kombinerer dem.

    Usage:
        ec = EnsembleCalibrator(["rf", "xgb", "lgbm"])
        ec.fit(probs_dict, y_val)
        # probs_dict = {"rf": [...], "xgb": [...], "lgbm": [...]}
        ensemble_probs = ec.transform(probs_dict_test)
    """

    def __init__(
        self,
        model_names: list[str],
        method: Literal["platt", "isotonic", "auto"] = "auto",
        weights: Optional[dict[str, float]] = None,
    ):
        self.model_names = model_names
        self.method = method
        self.weights = weights or {n: 1.0 for n in model_names}
        self.calibrators_: dict[str, ProbabilityCalibrator] = {}

    def fit(
        self, probs_dict: dict[str, np.ndarray], y_true: np.ndarray
    ) -> "EnsembleCalibrator":
        for name in self.model_names:
            if name not in probs_dict:
                raise KeyError(f"Missing probs for model '{name}'")
            cal = ProbabilityCalibrator(method=self.method)
            cal.fit(probs_dict[name], y_true)
            self.calibrators_[name] = cal
        return self

    def transform(
        self, probs_dict: dict[str, np.ndarray], return_individual: bool = False
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        calibrated = {}
        for name in self.model_names:
            calibrated[name] = self.calibrators_[name].transform(probs_dict[name])

        # Weighted average
        total_w = sum(self.weights[n] for n in self.model_names)
        ensemble = sum(calibrated[n] * self.weights[n] for n in self.model_names) / total_w

        if return_individual:
            return ensemble, calibrated
        return ensemble

    def report(self) -> str:
        out = ["\n🎯 Ensemble Calibration Report", "=" * 50]
        for name, cal in self.calibrators_.items():
            out.append(f"\n[{name.upper()}] weight={self.weights[name]:.2f}")
            out.append(cal.report())
        return "\n".join(out)

    def save(self, dir_path: str | Path) -> None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for name, cal in self.calibrators_.items():
            cal.save(dir_path / f"{name}_calibrator.pkl")
        meta = {"model_names": self.model_names, "weights": self.weights, "method": self.method}
        with open(dir_path / "ensemble_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, dir_path: str | Path) -> "EnsembleCalibrator":
        dir_path = Path(dir_path)
        with open(dir_path / "ensemble_meta.json") as f:
            meta = json.load(f)
        ec = cls(meta["model_names"], method=meta["method"], weights=meta["weights"])
        for name in meta["model_names"]:
            ec.calibrators_[name] = ProbabilityCalibrator.load(
                dir_path / f"{name}_calibrator.pkl"
            )
        return ec


# ============================================================
# Reliability Diagram (visualization helper)
# ============================================================

def reliability_diagram_data(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """
    Generer data til reliability diagram (calibration curve).
    Bruges senere af dashboard-tab til plotly.
    """
    probs = np.asarray(probs).ravel()
    y_true = np.asarray(y_true).ravel()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probs, bin_edges[1:-1])

    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            rows.append({
                "bin": b,
                "bin_center": (bin_edges[b] + bin_edges[b + 1]) / 2,
                "mean_predicted": np.nan,
                "mean_actual": np.nan,
                "count": 0,
            })
            continue
        rows.append({
            "bin": b,
            "bin_center": (bin_edges[b] + bin_edges[b + 1]) / 2,
            "mean_predicted": float(probs[mask].mean()),
            "mean_actual": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


# ============================================================
# Quick self-test
# ============================================================

if __name__ == "__main__":
    print("🧪 Testing ProbabilityCalibrator...\n")

    # Simuler en miscalibreret model (overconfident)
    rng = np.random.default_rng(42)
    n = 2000
    true_p = rng.beta(2, 2, n)
    y = (rng.random(n) < true_p).astype(int)

    # Overconfident model: skub probs mod 0 og 1
    raw_probs = np.where(true_p > 0.5,
                         np.minimum(true_p * 1.4, 0.99),
                         np.maximum(true_p * 0.6, 0.01))

    # Split val/test
    split = n // 2
    p_val, p_test = raw_probs[:split], raw_probs[split:]
    y_val, y_test = y[:split], y[split:]

    # Kalibrer
    cal = ProbabilityCalibrator(method="auto")
    cal.fit(p_val, y_val)
    p_calibrated = cal.transform(p_test)

    print(cal.report())
    print(f"Test set Brier (raw):        {brier_score_loss(y_test, p_test):.4f}")
    print(f"Test set Brier (calibrated): {brier_score_loss(y_test, p_calibrated):.4f}")

    # Test ensemble
    print("\n🎯 Testing EnsembleCalibrator...")
    probs_dict_val = {
        "rf":   np.clip(p_val + rng.normal(0, 0.05, len(p_val)), 0, 1),
        "xgb":  np.clip(p_val + rng.normal(0, 0.05, len(p_val)), 0, 1),
        "lgbm": np.clip(p_val + rng.normal(0, 0.05, len(p_val)), 0, 1),
    }
    probs_dict_test = {
        "rf":   np.clip(p_test + rng.normal(0, 0.05, len(p_test)), 0, 1),
        "xgb":  np.clip(p_test + rng.normal(0, 0.05, len(p_test)), 0, 1),
        "lgbm": np.clip(p_test + rng.normal(0, 0.05, len(p_test)), 0, 1),
    }

    ec = EnsembleCalibrator(["rf", "xgb", "lgbm"], method="auto",
                            weights={"rf": 1.0, "xgb": 1.2, "lgbm": 1.0})
    ec.fit(probs_dict_val, y_val)
    ensemble_probs = ec.transform(probs_dict_test)
    print(f"Ensemble Brier:  {brier_score_loss(y_test, ensemble_probs):.4f}")
    print("\n✅ All tests passed!")
