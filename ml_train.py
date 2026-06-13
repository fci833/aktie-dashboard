"""
ml_train.py - ML Training Pipeline (STREAMLIT CLOUD SAFE)
====================================
Trains ensemble of ML models (Random Forest, XGBoost, LightGBM) on 
backfilled training data. Saves trained models to disk.

Main entry: train_all_models(data_dict, asset_class="stock")

🔧 STREAMLIT CLOUD FIXES:
- LightGBM: n_jobs=1, force_col_wise=True (forhindrer deadlock)
- XGBoost: n_jobs=1 (sikker på 1-core systems)
- Cross-validation: n_jobs=1 (forhindrer nested parallelism)
- Reduceret n_estimators for hurtigere træning
"""
import warnings
warnings.filterwarnings("ignore")

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder

# Optional: XGBoost & LightGBM (graceful fallback)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# ==========================================
# CONFIG
# ==========================================

MODELS_DIR = Path("ml_models")
MODELS_DIR.mkdir(exist_ok=True)

CLASS_LABELS = ["SELL", "HOLD", "BUY"]

# 🔑 KRITISK: Streamlit Cloud har kun 1 CPU core (delt)
#    Brug n_jobs=1 i alle modeller + CV for at undgå deadlock
STREAMLIT_CLOUD_SAFE = True  # Sæt til False hvis du kører lokalt

# Auto-detect: sæt n_jobs til 1 hvis vi er i constrained env
if STREAMLIT_CLOUD_SAFE:
    MODEL_N_JOBS = 1
    CV_N_JOBS = 1
else:
    MODEL_N_JOBS = -1
    CV_N_JOBS = -1


# ==========================================
# MODEL FACTORY (PATCHED)
# ==========================================

def create_classifier(name: str, n_classes: int = 3):
    """Create a classifier by name. Streamlit Cloud safe."""
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=150,           # Reduceret fra 200
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=MODEL_N_JOBS,        # 🔑 SAFE
        )
    elif name == "xgboost" and HAS_XGBOOST:
        return xgb.XGBClassifier(
            n_estimators=150,           # Reduceret fra 200
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=MODEL_N_JOBS,        # 🔑 SAFE
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",         # Hurtigere på CPU
            verbosity=0,                # Stille
        )
    elif name == "lightgbm" and HAS_LIGHTGBM:
        return lgb.LGBMClassifier(
            n_estimators=100,           # Reduceret fra 200
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,                   # 🔑 KRITISK: ALTID 1 for LightGBM
            verbose=-1,
            force_col_wise=True,        # 🔑 Forhindrer threading-hang
            min_child_samples=20,
            deterministic=True,         # Reproducerbart
        )
    else:
        return None


def create_regressor(name: str):
    """Create a regressor by name. Streamlit Cloud safe."""
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=MODEL_N_JOBS,        # 🔑 SAFE
        )
    elif name == "xgboost" and HAS_XGBOOST:
        return xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=MODEL_N_JOBS,        # 🔑 SAFE
            tree_method="hist",
            verbosity=0,
        )
    elif name == "lightgbm" and HAS_LIGHTGBM:
        return lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,                   # 🔑 KRITISK: ALTID 1 for LightGBM
            verbose=-1,
            force_col_wise=True,        # 🔑 Forhindrer threading-hang
            min_child_samples=20,
            deterministic=True,
        )
    else:
        return None


def get_available_models() -> List[str]:
    """Return list of available models."""
    available = ["random_forest"]
    if HAS_XGBOOST:
        available.append("xgboost")
    if HAS_LIGHTGBM:
        available.append("lightgbm")
    return available


# ==========================================
# CLASSIFICATION TRAINING (PATCHED CV)
# ==========================================

def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest",
    test_size: float = 0.2,
    cv_folds: int = 3,                  # 🔑 Reduceret fra 5 til 3 for hastighed
) -> Dict:
    """
    Train a single classifier and return metrics + model.
    Streamlit Cloud safe (n_jobs=1 i CV).
    """
    # Encode labels (SELL=0, HOLD=1, BUY=2)
    le = LabelEncoder()
    le.fit(CLASS_LABELS)
    y_encoded = le.transform(y)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=test_size,
        random_state=42,
        stratify=y_encoded,
    )

    # Create model
    model = create_classifier(model_name)
    if model is None:
        return {"error": f"Model '{model_name}' not available"}

    # Train
    model.fit(X_train, y_train)

    # Test predictions
    y_pred = model.predict(X_test)

    # Cross-validation (SAFE: n_jobs=1)
    try:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X, y_encoded,
            cv=cv,
            scoring="f1_macro",
            n_jobs=CV_N_JOBS,           # 🔑 SAFE: 1 på Streamlit
        )
    except Exception as e:
        print(f"⚠️ CV failed for {model_name}: {e}")
        cv_scores = np.array([0.0])

    # Metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Per-class metrics
    try:
        per_class = {}
        for i, label in enumerate(le.classes_):
            mask = y_test == i
            if mask.sum() > 0:
                pred_class = (y_pred == i).sum()
                actual_class = mask.sum()
                correct = ((y_pred == i) & mask).sum()
                per_class[label] = {
                    "support": int(actual_class),
                    "precision": float(correct / pred_class) if pred_class > 0 else 0.0,
                    "recall": float(correct / actual_class) if actual_class > 0 else 0.0,
                }
        metrics["per_class"] = per_class
    except Exception:
        metrics["per_class"] = {}

    # Confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
    except Exception:
        metrics["confusion_matrix"] = []

    # Feature importance
    feature_importance = {}
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for feat, imp in zip(X.columns, importances):
                feature_importance[feat] = float(imp)
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
    except Exception:
        pass

    return {
        "model": model,
        "model_name": model_name,
        "label_encoder": le,
        "feature_columns": list(X.columns),
        "metrics": metrics,
        "feature_importance": feature_importance,
        "type": "classifier",
    }


# ==========================================
# REGRESSION TRAINING (PATCHED CV)
# ==========================================

def train_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest",
    test_size: float = 0.2,
    cv_folds: int = 3,                  # 🔑 Reduceret fra 5 til 3
) -> Dict:
    """Train a single regressor. Streamlit Cloud safe."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
    )

    model = create_regressor(model_name)
    if model is None:
        return {"error": f"Model '{model_name}' not available"}

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Cross-validation (SAFE: n_jobs=1)
    try:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=CV_N_JOBS,           # 🔑 SAFE
        )
        cv_mae = -cv_scores.mean()
    except Exception as e:
        print(f"⚠️ CV failed for {model_name}: {e}")
        cv_mae = 0.0

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "cv_mae": float(cv_mae),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }

    feature_importance = {}
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for feat, imp in zip(X.columns, importances):
                feature_importance[feat] = float(imp)
            feature_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
    except Exception:
        pass

    return {
        "model": model,
        "model_name": model_name,
        "feature_columns": list(X.columns),
        "metrics": metrics,
        "feature_importance": feature_importance,
        "type": "regressor",
    }


# ==========================================
# ENSEMBLE TRAINING
# ==========================================

def train_ensemble_for_horizon(
    data_dict: Dict,
    horizon: int,
    asset_class: str = "stock",
    progress_callback=None,
) -> Dict:
    """
    Train all available models for one time horizon (30d/90d/180d).
    
    Returns dict with all trained models + ensemble metrics.
    """
    X_clf_key = f"X_clf_{horizon}d"
    y_clf_key = f"y_clf_{horizon}d"
    X_reg_key = f"X_{horizon}d"
    y_reg_key = f"y_reg_{horizon}d"

    if X_clf_key not in data_dict or y_clf_key not in data_dict:
        return {"error": f"No data for {horizon}d horizon"}

    X_clf = data_dict[X_clf_key]
    y_clf = data_dict[y_clf_key]
    X_reg = data_dict[X_reg_key]
    y_reg = data_dict[y_reg_key]

    if len(X_clf) < 50:
        return {"error": f"Too few samples ({len(X_clf)}) for {horizon}d"}

    available = get_available_models()
    results = {
        "horizon": horizon,
        "asset_class": asset_class,
        "n_samples_clf": len(X_clf),
        "n_samples_reg": len(X_reg),
        "n_features": len(X_clf.columns),
        "feature_columns": list(X_clf.columns),
        "models_trained": [],
        "classifiers": {},
        "regressors": {},
        "trained_at": datetime.now().isoformat(),
    }

    n_steps = len(available) * 2  # classifier + regressor
    step = 0

    # Train classifiers
    for model_name in available:
        if progress_callback:
            progress_callback(step / n_steps, f"Træner {model_name} (klassifikation)...")
        step += 1

        try:
            result = train_classifier(X_clf, y_clf, model_name=model_name)
            if "error" not in result:
                results["classifiers"][model_name] = result
                results["models_trained"].append(f"{model_name}_clf")
        except Exception as e:
            print(f"⚠️ {model_name} classifier failed: {e}")

    # Train regressors
    for model_name in available:
        if progress_callback:
            progress_callback(step / n_steps, f"Træner {model_name} (regression)...")
        step += 1

        try:
            result = train_regressor(X_reg, y_reg, model_name=model_name)
            if "error" not in result:
                results["regressors"][model_name] = result
                results["models_trained"].append(f"{model_name}_reg")
        except Exception as e:
            print(f"⚠️ {model_name} regressor failed: {e}")

    if progress_callback:
        progress_callback(1.0, "Færdig!")

    return results


# ==========================================
# SAVE / LOAD MODELS
# ==========================================

def save_models(results: Dict, asset_class: str = "stock") -> Dict[str, str]:
    """
    Save trained models to disk as .joblib files.
    Returns dict of {model_key: filepath}
    """
    horizon = results.get("horizon", "unknown")
    saved = {}

    # Save classifiers
    for name, clf_data in results.get("classifiers", {}).items():
        filename = f"{asset_class}_{horizon}d_{name}_clf.joblib"
        filepath = MODELS_DIR / filename
        try:
            joblib.dump({
                "model": clf_data["model"],
                "label_encoder": clf_data.get("label_encoder"),
                "feature_columns": clf_data["feature_columns"],
                "metrics": clf_data["metrics"],
                "feature_importance": clf_data.get("feature_importance", {}),
                "type": "classifier",
                "model_name": name,
                "asset_class": asset_class,
                "horizon": horizon,
                "trained_at": results.get("trained_at"),
            }, filepath, compress=3)
            saved[f"{name}_clf"] = str(filepath)
        except Exception as e:
            print(f"⚠️ Could not save {name} classifier: {e}")

    # Save regressors
    for name, reg_data in results.get("regressors", {}).items():
        filename = f"{asset_class}_{horizon}d_{name}_reg.joblib"
        filepath = MODELS_DIR / filename
        try:
            joblib.dump({
                "model": reg_data["model"],
                "feature_columns": reg_data["feature_columns"],
                "metrics": reg_data["metrics"],
                "feature_importance": reg_data.get("feature_importance", {}),
                "type": "regressor",
                "model_name": name,
                "asset_class": asset_class,
                "horizon": horizon,
                "trained_at": results.get("trained_at"),
            }, filepath, compress=3)
            saved[f"{name}_reg"] = str(filepath)
        except Exception as e:
            print(f"⚠️ Could not save {name} regressor: {e}")

    # Save metadata file
    meta_path = MODELS_DIR / f"{asset_class}_{horizon}d_meta.json"
    try:
        meta = {
            "asset_class": asset_class,
            "horizon": horizon,
            "n_samples_clf": results.get("n_samples_clf"),
            "n_samples_reg": results.get("n_samples_reg"),
            "n_features": results.get("n_features"),
            "feature_columns": results.get("feature_columns"),
            "models_trained": results.get("models_trained", []),
            "trained_at": results.get("trained_at"),
            "saved_files": list(saved.keys()),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        saved["meta"] = str(meta_path)
    except Exception as e:
        print(f"⚠️ Could not save metadata: {e}")

    return saved


def load_model(asset_class: str, horizon: int, model_name: str, model_type: str = "clf") -> Optional[Dict]:
    """Load a single trained model."""
    filename = f"{asset_class}_{horizon}d_{model_name}_{model_type}.joblib"
    filepath = MODELS_DIR / filename
    if not filepath.exists():
        return None
    try:
        return joblib.load(filepath)
    except Exception as e:
        print(f"⚠️ Could not load {filepath}: {e}")
        return None


def list_saved_models() -> List[Dict]:
    """List all saved models."""
    models = []
    for filepath in sorted(MODELS_DIR.glob("*.joblib")):
        try:
            parts = filepath.stem.split("_")
            if len(parts) >= 4:
                asset_class = parts[0]
                horizon_str = parts[1].replace("d", "")
                model_type = parts[-1]
                model_name = "_".join(parts[2:-1])
                models.append({
                    "filename": filepath.name,
                    "filepath": str(filepath),
                    "asset_class": asset_class,
                    "horizon": int(horizon_str) if horizon_str.isdigit() else 0,
                    "model_name": model_name,
                    "type": model_type,
                    "size_kb": filepath.stat().st_size / 1024,
                })
        except Exception:
            continue
    return models


def get_training_summary() -> Dict:
    """Quick stats about saved models."""
    models = list_saved_models()
    summary = {
        "total_models": len(models),
        "by_asset_class": {},
        "by_horizon": {},
        "by_model": {},
    }
    for m in models:
        ac = m["asset_class"]
        h = m["horizon"]
        mn = m["model_name"]
        summary["by_asset_class"][ac] = summary["by_asset_class"].get(ac, 0) + 1
        summary["by_horizon"][h] = summary["by_horizon"].get(h, 0) + 1
        summary["by_model"][mn] = summary["by_model"].get(mn, 0) + 1
    return summary


# ==========================================
# MAIN ENTRY POINT
# ==========================================

def train_all_models(
    data_dict: Dict,
    asset_class: str = "stock",
    horizons: List[int] = [30, 90, 180],
    save: bool = True,
    progress_callback=None,
) -> Dict:
    """
    Train models for ALL horizons.

    Args:
        data_dict: Output from ml_data.get_training_data()
        asset_class: "stock" or "crypto"
        horizons: List of horizons to train
        save: Whether to save models to disk
        progress_callback: callable(progress_pct, status_text)

    Returns dict with full training results.
    """
    if "error" in data_dict:
        return {"error": data_dict["error"]}

    available = get_available_models()
    if not available:
        return {"error": "No ML models available! Install scikit-learn at minimum."}

    results = {
        "asset_class": asset_class,
        "available_models": available,
        "horizons_trained": [],
        "results_per_horizon": {},
        "saved_files": {},
        "started_at": datetime.now().isoformat(),
    }

    n_horizons = len(horizons)

    for i, h in enumerate(horizons):
        if progress_callback:
            base_progress = i / n_horizons

            def horizon_cb(pct, text, _h=h, _base=base_progress, _n=n_horizons):
                overall = _base + (pct / _n)
                progress_callback(overall, f"[{_h}d] {text}")
        else:
            horizon_cb = None

        h_result = train_ensemble_for_horizon(
            data_dict, h, asset_class, progress_callback=horizon_cb
        )

        if "error" not in h_result:
            results["results_per_horizon"][h] = h_result
            results["horizons_trained"].append(h)

            if save:
                saved = save_models(h_result, asset_class)
                results["saved_files"][h] = saved
        else:
            results["results_per_horizon"][h] = h_result

    results["completed_at"] = datetime.now().isoformat()
    results["n_horizons_trained"] = len(results["horizons_trained"])

    return results


# ==========================================
# CLI TEST
# ==========================================

if __name__ == "__main__":
    print("=" * 70)
    print("ML TRAIN - TEST (Streamlit Cloud Safe)")
    print("=" * 70)
    print(f"Available models: {get_available_models()}")
    print(f"  - Random Forest: ✅ (n_jobs={MODEL_N_JOBS})")
    print(f"  - XGBoost: {'✅' if HAS_XGBOOST else '❌'} (n_jobs={MODEL_N_JOBS})")
    print(f"  - LightGBM: {'✅' if HAS_LIGHTGBM else '❌'} (n_jobs=1, force_col_wise=True)")
    print(f"  - CV n_jobs: {CV_N_JOBS}")

    try:
        from ml_data import get_training_data
        print("\n📊 Loading training data...")
        data = get_training_data(asset_class="stock", verbose=False)

        if "error" in data:
            print(f"❌ {data['error']}")
        else:
            print(f"✅ Loaded {data.get('n_samples_30d', 0)} samples for 30d")
            print("\n🚀 Starting training...")

            def cb(pct, text):
                print(f"  [{pct*100:.0f}%] {text}")

            results = train_all_models(
                data, asset_class="stock",
                horizons=[30],
                save=True,
                progress_callback=cb,
            )

            print(f"\n✅ Trained {results['n_horizons_trained']} horizons")
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
