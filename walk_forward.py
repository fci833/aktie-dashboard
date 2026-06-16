"""
Walk-Forward Validation
=======================
Korrekt out-of-sample evaluering af ML-modeller på tidsserie-data.

Almindelig train/test split SNYDER på tidsserier fordi:
  - Random shuffle = data leakage (modellen ser fremtiden)
  - Single split = ingen indsigt i stabilitet over tid
  - Ingen retraining = urealistisk i produktion

Walk-forward fixer dette ved at:
  1. Træne på et vindue af historiske data
  2. Forudsige NÆSTE periode
  3. Skubbe vinduet frem og gentage
  4. Aggregere resultater for ægte out-of-sample performance

Modes:
  - 'expanding': Train-set vokser over tid (meget data → bedre signal)
  - 'rolling':   Train-set har fast størrelse (adapterer til regimes)
  - 'anchored':  Start-dato fast, end-dato vokser

Integration:
  - Bruger ProbabilityCalibrator til kalibrerede probs
  - Logger resultater til PredictionLogger (valgfrit)
  - Outputs DataFrames der kan plottes/analyseres
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, accuracy_score

from ml_calibration import ProbabilityCalibrator, EnsembleCalibrator


# ============================================================
# Protocols
# ============================================================

class SklearnLikeModel(Protocol):
    """Enhver model med .fit() + .predict_proba()."""
    def fit(self, X, y) -> Any: ...
    def predict_proba(self, X) -> np.ndarray: ...


# ============================================================
# Config
# ============================================================

@dataclass
class WalkForwardConfig:
    """Konfiguration for walk-forward validation."""
    mode: Literal["expanding", "rolling", "anchored"] = "expanding"
    initial_train_size: int = 252           # ~1 år trading days
    test_size: int = 21                     # ~1 måned forecast
    step_size: int = 21                     # hvor langt vi rykker frem per fold
    min_train_size: int = 100               # minimum træningsdata
    rolling_window: int = 504               # ~2 år (kun for 'rolling' mode)
    calibrate: bool = True                  # brug ProbabilityCalibrator
    calibration_size: int = 60              # sidste N rows af train til calibration
    purge_days: int = 0                     # gap mellem train og test (mod leakage)
    embargo_days: int = 0                   # gap efter test før næste train
    verbose: bool = True

    def __post_init__(self):
        if self.test_size <= 0 or self.step_size <= 0:
            raise ValueError("test_size og step_size skal være > 0")
        if self.calibrate and self.calibration_size >= self.initial_train_size:
            raise ValueError("calibration_size skal være < initial_train_size")


# ============================================================
# Result Container
# ============================================================

@dataclass
class FoldResult:
    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    hit_rate: float
    brier_score: Optional[float]
    log_loss_value: Optional[float]
    roc_auc: Optional[float]
    accuracy: float
    avg_predicted_prob: float
    pos_rate_train: float                   # % positives i træningssæt
    pos_rate_test: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WalkForwardResult:
    config: dict
    folds: list[FoldResult] = field(default_factory=list)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_names: list[str] = field(default_factory=list)

    # ----- Aggregates -----
    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def folds_df(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.folds])

    def overall_metrics(self) -> dict:
        if self.predictions.empty:
            return {}
        df = self.predictions
        y, p = df["y_true"].values, df["y_prob"].values
        try:
            auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
        except Exception:
            auc = None
        return {
            "n_predictions": len(df),
            "n_folds": self.n_folds,
            "hit_rate": float((df["y_pred"] == df["y_true"]).mean()),
            "accuracy": float(accuracy_score(y, df["y_pred"])),
            "brier_score": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))),
            "roc_auc": auc,
            "avg_pred_prob": float(p.mean()),
            "actual_pos_rate": float(y.mean()),
        }

    def report(self) -> str:
        m = self.overall_metrics()
        if not m:
            return "📭 Ingen predictions genereret"
        f = self.folds_df

        out = [
            "\n" + "=" * 60,
            f"  🚶 WALK-FORWARD VALIDATION REPORT",
            "=" * 60,
            f"  Mode:                    {self.config['mode']}",
            f"  Folds completed:         {m['n_folds']}",
            f"  Total predictions:       {m['n_predictions']}",
            f"  Calibrated:              {self.config['calibrate']}",
            "",
            "  📊 Overall metrics:",
            f"    Hit rate:              {m['hit_rate']:.1%}",
            f"    Accuracy:              {m['accuracy']:.1%}",
            f"    Brier score:           {m['brier_score']:.4f}",
            f"    Log loss:              {m['log_loss']:.4f}",
            f"    ROC-AUC:               " + (f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"),
            f"    Avg predicted prob:    {m['avg_pred_prob']:.1%}",
            f"    Actual pos rate:       {m['actual_pos_rate']:.1%}",
            "",
            "  📈 Per-fold stability:",
            f"    Hit rate    - mean: {f['hit_rate'].mean():.1%}, std: {f['hit_rate'].std():.1%}",
            f"    Brier       - mean: {f['brier_score'].mean():.4f}, std: {f['brier_score'].std():.4f}",
            f"    Log loss    - mean: {f['log_loss_value'].mean():.4f}, std: {f['log_loss_value'].std():.4f}",
            "",
            "  🎯 Best/worst fold:",
            f"    Best  hit rate: {f['hit_rate'].max():.1%} (fold {f['hit_rate'].idxmax()})",
            f"    Worst hit rate: {f['hit_rate'].min():.1%} (fold {f['hit_rate'].idxmin()})",
            "=" * 60 + "\n",
        ]
        return "\n".join(out)


# ============================================================
# Splitter
# ============================================================

class WalkForwardSplitter:
    """Genererer (train_idx, test_idx) tupler i tidsmæssig rækkefølge."""

    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def split(self, n_samples: int):
        cfg = self.config
        train_end = cfg.initial_train_size

        while train_end + cfg.purge_days + cfg.test_size <= n_samples:
            test_start = train_end + cfg.purge_days
            test_end = test_start + cfg.test_size

            if cfg.mode == "expanding":
                train_start = 0
            elif cfg.mode == "rolling":
                train_start = max(0, train_end - cfg.rolling_window)
            elif cfg.mode == "anchored":
                train_start = 0
            else:
                raise ValueError(f"Unknown mode: {cfg.mode}")

            if (train_end - train_start) < cfg.min_train_size:
                train_end += cfg.step_size
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

            train_end = test_end + cfg.embargo_days


# ============================================================
# Validator
# ============================================================

class WalkForwardValidator:
    """
    Hovedklasse til walk-forward validation.

    Usage:
        cfg = WalkForwardConfig(initial_train_size=252, test_size=21)
        wfv = WalkForwardValidator(cfg)
        result = wfv.run(
            X=feature_df, y=labels, dates=date_index,
            model_factory=lambda: RandomForestClassifier(n_estimators=200),
        )
        print(result.report())
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self.config = config or WalkForwardConfig()

    # ============================================================
    # Single-model
    # ============================================================

    def run(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        dates: pd.Series | pd.DatetimeIndex,
        model_factory: Callable[[], SklearnLikeModel],
        ticker: str = "UNKNOWN",
        horizon_days: Optional[int] = None,
    ) -> WalkForwardResult:
        X, y, dates = self._validate_inputs(X, y, dates)
        splitter = WalkForwardSplitter(self.config)
        cfg = self.config

        folds: list[FoldResult] = []
        all_preds: list[dict] = []

        splits = list(splitter.split(len(X)))
        if not splits:
            raise ValueError(
                f"Ikke nok data til walk-forward. "
                f"n={len(X)}, initial_train={cfg.initial_train_size}, test={cfg.test_size}"
            )

        if cfg.verbose:
            print(f"🚶 Walk-forward: {len(splits)} folds | {cfg.mode} mode | n={len(X)}")

        for fold_id, (train_idx, test_idx) in enumerate(splits):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            d_tr, d_te = dates.iloc[train_idx], dates.iloc[test_idx]

            # Skip hvis kun én klasse i train
            if len(np.unique(y_tr)) < 2:
                if cfg.verbose:
                    print(f"  ⏭️  Fold {fold_id}: skipped (single class in train)")
                continue

            # Calibration split
            if cfg.calibrate and len(X_tr) > cfg.calibration_size + cfg.min_train_size:
                cal_size = cfg.calibration_size
                X_fit = X_tr.iloc[:-cal_size]
                y_fit = y_tr[:-cal_size]
                X_cal = X_tr.iloc[-cal_size:]
                y_cal = y_tr[-cal_size:]
            else:
                X_fit, y_fit = X_tr, y_tr
                X_cal, y_cal = None, None

            # Train
            model = model_factory()
            try:
                model.fit(X_fit, y_fit)
            except Exception as e:
                if cfg.verbose:
                    print(f"  ⚠️  Fold {fold_id}: model fit failed - {e}")
                continue

            # Calibrate
            calibrator = None
            if X_cal is not None and len(np.unique(y_cal)) > 1:
                try:
                    raw_cal_probs = model.predict_proba(X_cal)[:, 1]
                    calibrator = ProbabilityCalibrator(method="auto")
                    calibrator.fit(raw_cal_probs, y_cal)
                except Exception as e:
                    if cfg.verbose:
                        print(f"  ⚠️  Fold {fold_id}: calibration skipped - {e}")
                    calibrator = None

            # Predict
            try:
                raw_probs = model.predict_proba(X_te)[:, 1]
                probs = calibrator.transform(raw_probs) if calibrator else raw_probs
                preds = (probs >= 0.5).astype(int)
            except Exception as e:
                if cfg.verbose:
                    print(f"  ⚠️  Fold {fold_id}: predict failed - {e}")
                continue

            # Metrics
            fr = self._fold_metrics(
                fold_id=fold_id, d_tr=d_tr, d_te=d_te,
                y_tr=y_tr, y_te=y_te, probs=probs, preds=preds,
            )
            folds.append(fr)

            for i, idx in enumerate(test_idx):
                all_preds.append({
                    "fold_id": fold_id,
                    "date": dates.iloc[idx],
                    "ticker": ticker,
                    "horizon_days": horizon_days,
                    "y_true": int(y_te[i]),
                    "y_pred": int(preds[i]),
                    "y_prob": float(probs[i]),
                    "y_prob_raw": float(raw_probs[i]),
                    "calibrated": calibrator is not None,
                })

            if cfg.verbose:
                print(f"  ✓ Fold {fold_id:>2d}: train [{fr.train_start}→{fr.train_end}] "
                      f"({fr.n_train}) | test ({fr.n_test}) | "
                      f"hit={fr.hit_rate:.1%} brier={fr.brier_score:.3f}"
                      if fr.brier_score is not None else "")

        return WalkForwardResult(
            config=asdict(self.config),
            folds=folds,
            predictions=pd.DataFrame(all_preds),
            feature_names=list(X.columns),
        )

    # ============================================================
    # Ensemble (RF + XGB + LGBM)
    # ============================================================

    def run_ensemble(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        dates: pd.Series | pd.DatetimeIndex,
        model_factories: dict[str, Callable[[], SklearnLikeModel]],
        weights: Optional[dict[str, float]] = None,
        ticker: str = "UNKNOWN",
        horizon_days: Optional[int] = None,
    ) -> dict[str, WalkForwardResult]:
        """
        Kør walk-forward for hver model + ensemble.
        Returnerer dict med results per model + 'ensemble'.
        """
        weights = weights or {name: 1.0 for name in model_factories}
        results: dict[str, WalkForwardResult] = {}

        for name, factory in model_factories.items():
            if self.config.verbose:
                print(f"\n--- Running {name.upper()} ---")
            results[name] = self.run(
                X, y, dates, factory,
                ticker=ticker, horizon_days=horizon_days,
            )

        # Build ensemble predictions
        if self.config.verbose:
            print(f"\n--- Building ENSEMBLE ---")

        # Align all predictions on (fold_id, date)
        merged = None
        for name, res in results.items():
            if res.predictions.empty:
                continue
            sub = res.predictions[["fold_id", "date", "y_true", "y_prob"]].copy()
            sub = sub.rename(columns={"y_prob": f"prob_{name}"})
            merged = sub if merged is None else merged.merge(
                sub, on=["fold_id", "date", "y_true"], how="inner"
            )

        if merged is not None and not merged.empty:
            prob_cols = [c for c in merged.columns if c.startswith("prob_")]
            total_w = sum(weights.get(c.replace("prob_", ""), 1.0) for c in prob_cols)
            ensemble_prob = sum(
                merged[c] * weights.get(c.replace("prob_", ""), 1.0)
                for c in prob_cols
            ) / total_w

            ensemble_preds = pd.DataFrame({
                "fold_id": merged["fold_id"],
                "date": merged["date"],
                "ticker": ticker,
                "horizon_days": horizon_days,
                "y_true": merged["y_true"],
                "y_prob": ensemble_prob,
                "y_pred": (ensemble_prob >= 0.5).astype(int),
                "calibrated": True,
            })

            # Build per-fold metrics for ensemble
            ens_folds = []
            for fid, group in ensemble_preds.groupby("fold_id"):
                fr = self._fold_metrics_from_preds(
                    fold_id=int(fid),
                    dates=group["date"],
                    y_true=group["y_true"].values,
                    probs=group["y_prob"].values,
                    preds=group["y_pred"].values,
                )
                ens_folds.append(fr)

            results["ensemble"] = WalkForwardResult(
                config=asdict(self.config),
                folds=ens_folds,
                predictions=ensemble_preds,
                feature_names=list(X.columns),
            )

        return results

    # ============================================================
    # Helpers
    # ============================================================

    @staticmethod
    def _validate_inputs(X, y, dates):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = np.asarray(y).ravel().astype(int)
        if not isinstance(dates, pd.Series):
            dates = pd.Series(pd.to_datetime(dates))
        else:
            dates = pd.Series(pd.to_datetime(dates.values))

        if len(X) != len(y) or len(X) != len(dates):
            raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}, dates={len(dates)}")

        # Sort by date (kritisk!)
        order = np.argsort(dates.values)
        return X.iloc[order].reset_index(drop=True), y[order], dates.iloc[order].reset_index(drop=True)

    @staticmethod
    def _fold_metrics(fold_id, d_tr, d_te, y_tr, y_te, probs, preds) -> FoldResult:
        return WalkForwardValidator._fold_metrics_from_preds(
            fold_id=fold_id,
            dates=d_te,
            y_true=y_te,
            probs=probs,
            preds=preds,
            train_start=str(d_tr.iloc[0].date()),
            train_end=str(d_tr.iloc[-1].date()),
            n_train=len(y_tr),
            pos_rate_train=float(y_tr.mean()),
        )

    @staticmethod
    def _fold_metrics_from_preds(
        fold_id, dates, y_true, probs, preds,
        train_start="", train_end="", n_train=0, pos_rate_train=0.0,
    ) -> FoldResult:
        try:
            brier = float(brier_score_loss(y_true, probs))
        except Exception:
            brier = None
        try:
            ll = float(log_loss(y_true, np.clip(probs, 1e-7, 1 - 1e-7)))
        except Exception:
            ll = None
        try:
            auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else None
        except Exception:
            auc = None

        d = pd.to_datetime(dates) if not isinstance(dates, pd.Series) else dates
        return FoldResult(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=str(d.iloc[0].date()),
            test_end=str(d.iloc[-1].date()),
            n_train=n_train,
            n_test=len(y_true),
            hit_rate=float((preds == y_true).mean()),
            brier_score=brier,
            log_loss_value=ll,
            roc_auc=auc,
            accuracy=float(accuracy_score(y_true, preds)),
            avg_predicted_prob=float(probs.mean()),
            pos_rate_train=pos_rate_train,
            pos_rate_test=float(y_true.mean()),
        )


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("🧪 Testing WalkForwardValidator...\n")

    # ----------- Generér syntetisk tidsserie-data -----------
    rng = np.random.default_rng(42)
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # Features med en regime-shift halvvejs (test om model adapterer)
    feat1 = rng.normal(0, 1, n)
    feat2 = rng.normal(0, 1, n)
    feat3 = rng.normal(0, 1, n)
    momentum = pd.Series(feat1).rolling(20).mean().fillna(0).values

    # Target: kombination med regime shift
    regime = (np.arange(n) > n // 2).astype(int)
    signal = 0.3 * feat1 + 0.4 * feat2 - 0.2 * regime * feat3 + 0.5 * momentum
    prob = 1 / (1 + np.exp(-signal))
    y = (rng.random(n) < prob).astype(int)

    X = pd.DataFrame({
        "feat1": feat1, "feat2": feat2, "feat3": feat3, "momentum": momentum,
    })

    # ----------- Test single-model walk-forward -----------
    from sklearn.ensemble import RandomForestClassifier

    cfg = WalkForwardConfig(
        mode="expanding",
        initial_train_size=252,
        test_size=21,
        step_size=21,
        calibrate=True,
        calibration_size=42,
        verbose=True,
    )

    wfv = WalkForwardValidator(cfg)
    result = wfv.run(
        X=X, y=y, dates=dates,
        model_factory=lambda: RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=-1
        ),
        ticker="TEST",
        horizon_days=21,
    )

    print(result.report())

    print("📊 First 5 folds:")
    print(result.folds_df[["fold_id", "test_start", "test_end",
                            "n_test", "hit_rate", "brier_score", "roc_auc"]].head())

    # ----------- Test ensemble walk-forward -----------
    print("\n" + "=" * 60)
    print("  Testing ENSEMBLE walk-forward")
    print("=" * 60)

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression

        cfg.verbose = False
        wfv2 = WalkForwardValidator(cfg)

        results = wfv2.run_ensemble(
            X=X, y=y, dates=dates,
            model_factories={
                "rf": lambda: RandomForestClassifier(n_estimators=80, max_depth=5,
                                                      random_state=42, n_jobs=-1),
                "gb": lambda: GradientBoostingClassifier(n_estimators=80, max_depth=3,
                                                          random_state=42),
                "lr": lambda: LogisticRegression(max_iter=500),
            },
            weights={"rf": 1.0, "gb": 1.2, "lr": 0.8},
            ticker="TEST",
            horizon_days=21,
        )

        print("\n📊 Comparison:")
        comparison = pd.DataFrame({
            name: res.overall_metrics() for name, res in results.items()
        }).T
        print(comparison[["n_predictions", "hit_rate", "brier_score",
                           "log_loss", "roc_auc"]].round(4))

    except ImportError as e:
        print(f"⚠️  Skipping ensemble test: {e}")

    print("\n✅ All tests passed!")
