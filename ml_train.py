"""
ml_train.py - ML Training Pipeline (STREAMLIT CLOUD OPTIMIZED v2)
====================================
Trains ensemble of ML models (Random Forest, XGBoost, LightGBM) on 
backfilled training data. Saves trained models to disk.

Main entry: train_all_models(data_dict, asset_class="stock")

🔧 v2 FIXES:
- Memory-optimized models for Streamlit Cloud (1GB RAM limit)
- VISIBLE error reporting (no more silent failures!)
- Garbage collection between models
- Reduced complexity for RF + XGBoost
- Better class balancing
"""
import warnings
warnings.filterwarnings("ignore")

import os
import gc                              # 🆕 For memory cleanup
import json
import joblib
import traceback                       # 🆕 For visible errors
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
from sklearn.utils.class_weight import compute_class_weight  # 🆕

# Optional: XGBoost & LightGBM
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

# 🔑 Streamlit Cloud safe mode
STREAMLIT_CLOUD_SAFE = True
MODEL_N_JOBS = 1 if STREAMLIT_CLOUD_SAFE else -1
CV_N_JOBS = 1 if STREAMLIT_CLOUD_SAFE else -1


# ==========================================
# MODEL FACTORY (MEMORY OPTIMIZED v2)
# ==========================================

def create_classifier(name: str, n_classes: int = 3):
    """
    Create a classifier — MEMORY OPTIMIZED for Streamlit Cloud.
    
    🆕 v2 changes:
    - RF: reduced n_estimators (150→80), max_depth (10→8)
    - XGBoost: reduced n_estimators (150→100), max_depth (6→5)
    - LightGBM: kept (already efficient)
    """
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=80,            # 🆕 Reduceret: 150→80 (saves ~50% RAM)
            max_depth=8,                # 🆕 Reduceret: 10→8 (mindre dybe træer)
            min_samples_split=20,       # 🆕 Øget: mere generalisering
            min_samples_leaf=10,        # 🆕 Øget: mindre overfit
            max_features="sqrt",        # 🆕 Eksplicit: bruger √44 ≈ 6 features
            class_weight="balanced",    # ⚖️ Hjælper HOLD-klassen
            random_state=42,
            n_jobs=MODEL_N_JOBS,
            bootstrap=True,             # Default
            oob_score=False,            # 🆕 Spar memory
            warm_start=False,           # 🆕 Spar memory
        )
    elif name == "xgboost" and HAS_XGBOOST:
        return xgb.XGBClassifier(
            n_estimators=100,           # 🆕 Reduceret: 150→100
            max_depth=5,                # 🆕 Reduceret: 6→5
            learning_rate=0.08,         # 🆕 Lidt højere for hurtigere convergens
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,         # 🆕 Mindre overfit
            gamma=0.1,                  # 🆕 Regularization
            random_state=42,
            n_jobs=MODEL_N_JOBS,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0,
            objective="multi:softprob", # Sikker default
        )
    elif name == "lightgbm" and HAS_LIGHTGBM:
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",    # ⚖️ Hjælper HOLD-klassen
            random_state=42,
            n_jobs=1,
            verbose=-1,
            force_col_wise=True,
            min_child_samples=20,
            deterministic=True,
        )
    else:
        return None


def create_regressor(name: str):
    """Create a regressor — MEMORY OPTIMIZED for Streamlit Cloud."""
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=80,            # 🆕 Reduceret
            max_depth=8,                # 🆕 Reduceret
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=MODEL_N_JOBS,
            bootstrap=True,
            oob_score=False,
            warm_start=False,
        )
    elif name == "xgboost" and HAS_XGBOOST:
        return xgb.XGBRegressor(
            n_estimators=100,           # 🆕 Reduceret
            max_depth=5,                # 🆕 Reduceret
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            random_state=42,
            n_jobs=MODEL_N_JOBS,
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
            n_jobs=1,
            verbose=-1,
            force_col_wise=True,
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
# CLASSIFICATION TRAINING (v2 — VISIBLE ERRORS)
# ==========================================

def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest",
    test_size: float = 0.2,
    cv_folds: int = 3,
) -> Dict:
    """
    Train a single classifier.
    
    🆕 v2: Returns errors visibly instead of silent fail.
    """
    try:
        # Encode labels
        le = LabelEncoder()
        le.fit(CLASS_LABELS)
        y_encoded = le.transform(y)

        # Print class distribution for debugging
        unique, counts = np.unique(y_encoded, return_counts=True)
        class_dist = dict(zip([CLASS_LABELS[i] for i in unique], counts))
        print(f"  📊 [{model_name}] Class distribution: {class_dist}")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=test_size,
            random_state=42,
            stratify=y_encoded,
        )

        # Create model
        model = create_classifier(model_name)
        if model is None:
            return {"error": f"Model '{model_name}' not available", "model_name": model_name}

        print(f"  🤖 [{model_name}] Training on {len(X_train)} samples × {X_train.shape[1]} features...")
        
        # Train
        model.fit(X_train, y_train)
        
        print(f"  ✅ [{model_name}] Training done!")

        # Test predictions
        y_pred = model.predict(X_test)

        # Cross-validation
        try:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                model, X, y_encoded,
                cv=cv,
                scoring="f1_macro",
                n_jobs=CV_N_JOBS,
            )
            print(f"  📈 [{model_name}] CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        except Exception as e:
            print(f"  ⚠️ [{model_name}] CV failed: {e}")
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

    except MemoryError as e:
        # 🆕 Eksplicit memory-fejl
        error_msg = f"MEMORY ERROR for {model_name}: not enough RAM. Try reducing n_estimators."
        print(f"  ❌ {error_msg}")
        return {"error": error_msg, "model_name": model_name, "error_type": "memory"}
    
    except Exception as e:
        # 🆕 Vis FULDE error message
        error_msg = f"FAILED for {model_name}: {type(e).__name__}: {str(e)}"
        print(f"  ❌ {error_msg}")
        print(f"     Traceback:")
        traceback.print_exc()
        return {"error": error_msg, "model_name": model_name, "error_type": type(e).__name__}


# ==========================================
# REGRESSION TRAINING (v2 — VISIBLE ERRORS)
# ==========================================

def train_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "random_forest",
    test_size: float = 0.2,
    cv_folds: int = 3,
) -> Dict:
    """Train a single regressor with visible error handling."""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
        )

        model = create_regressor(model_name)
        if model is None:
            return {"error": f"Model '{model_name}' not available", "model_name": model_name}

        print(f"  🤖 [{model_name}-reg] Training on {len(X_train)} samples...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        print(f"  ✅ [{model_name}-reg] Training done!")

        # Cross-validation
        try:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                model, X, y,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=CV_N_JOBS,
            )
            cv_mae = -cv_scores.mean()
        except Exception as e:
            print(f"  ⚠️ [{model_name}-reg] CV failed: {e}")
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

    except MemoryError as e:
        error_msg = f"MEMORY ERROR for {model_name}-reg: not enough RAM."
        print(f"  ❌ {error_msg}")
        return {"error": error_msg, "model_name": model_name, "error_type": "memory"}
    
    except Exception as e:
        error_msg = f"FAILED for {model_name}-reg: {type(e).__name__}: {str(e)}"
        print(f"  ❌ {error_msg}")
        traceback.print_exc()
        return {"error": error_msg, "model_name": model_name, "error_type": type(e).__name__}


# ==========================================
# ENSEMBLE TRAINING (v2 — WITH ERROR REPORTING)
# ==========================================

def train_ensemble_for_horizon(
    data_dict: Dict,
    horizon: int,
    asset_class: str = "stock",
    progress_callback=None,
) -> Dict:
    """
    Train all available models for one time horizon (30d/90d/180d).
    
    🆕 v2: 
    - Reports failed models explicitly
    - Garbage collection between models (saves RAM)
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
        "models_failed": [],          # 🆕 Track failures
        "failure_reasons": {},        # 🆕 Track why
        "classifiers": {},
        "regressors": {},
        "trained_at": datetime.now().isoformat(),
    }

    n_steps = len(available) * 2
    step = 0

    print(f"\n{'='*60}")
    print(f"🎯 Training horizon: {horizon}d")
    print(f"   Samples: {len(X_clf)} | Features: {len(X_clf.columns)}")
    print(f"   Available models: {available}")
    print(f"{'='*60}")

    # ==================== CLASSIFIERS ====================
    for model_name in available:
        if progress_callback:
            progress_callback(step / n_steps, f"Træner {model_name} (klassifikation)...")
        step += 1

        print(f"\n🤖 Training classifier: {model_name}")
        result = train_classifier(X_clf, y_clf, model_name=model_name)
        
        if "error" not in result:
            results["classifiers"][model_name] = result
            results["models_trained"].append(f"{model_name}_clf")
            print(f"   ✅ Saved {model_name} classifier")
        else:
            results["models_failed"].append(f"{model_name}_clf")
            results["failure_reasons"][f"{model_name}_clf"] = result["error"]
            print(f"   ❌ FAILED: {result['error']}")
        
        # 🆕 Memory cleanup mellem modeller
        gc.collect()

    # ==================== REGRESSORS ====================
    for model_name in available:
        if progress_callback:
            progress_callback(step / n_steps, f"Træner {model_name} (regression)...")
        step += 1

        print(f"\n🤖 Training regressor: {model_name}")
        result = train_regressor(X_reg, y_reg, model_name=model_name)
        
        if "error" not in result:
            results["regressors"][model_name] = result
            results["models_trained"].append(f"{model_name}_reg")
            print(f"   ✅ Saved {model_name} regressor")
        else:
            results["models_failed"].append(f"{model_name}_reg")
            results["failure_reasons"][f"{model_name}_reg"] = result["error"]
            print(f"   ❌ FAILED: {result['error']}")
        
        # 🆕 Memory cleanup mellem modeller
        gc.collect()

    if progress_callback:
        progress_callback(1.0, "Færdig!")

    # 🆕 Final summary
    print(f"\n{'='*60}")
    print(f"📊 Horizon {horizon}d summary:")
    print(f"   ✅ Trained: {len(results['models_trained'])}")
    print(f"   ❌ Failed: {len(results['models_failed'])}")
    if results["models_failed"]:
        print(f"   Failures:")
        for mname, reason in results["failure_reasons"].items():
            print(f"      - {mname}: {reason}")
    print(f"{'='*60}\n")

    return results


# ==========================================
# SAVE / LOAD MODELS (UNCHANGED)
# ==========================================

def save_models(results: Dict, asset_class: str = "stock") -> Dict[str, str]:
    """Save trained models to disk as .joblib files."""
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
            print(f"   💾 Saved: {filename}")
        except Exception as e:
            print(f"   ⚠️ Could not save {name} classifier: {e}")

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
            print(f"   💾 Saved: {filename}")
        except Exception as e:
            print(f"   ⚠️ Could not save {name} regressor: {e}")

    # Save metadata
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
            "models_failed": results.get("models_failed", []),  # 🆕
            "failure_reasons": results.get("failure_reasons", {}),  # 🆕
            "trained_at": results.get("trained_at"),
            "saved_files": list(saved.keys()),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        saved["meta"] = str(meta_path)
    except Exception as e:
        print(f"   ⚠️ Could not save metadata: {e}")

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
# MAIN ENTRY POINT (v2)
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
    
    🆕 v2: Better error tracking + memory cleanup.
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
        "horizons_failed": [],            # 🆕
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
            results["horizons_failed"].append(h)
            results["results_per_horizon"][h] = h_result

        # 🆕 Cleanup mellem horisonter
        gc.collect()

    results["completed_at"] = datetime.now().isoformat()
    results["n_horizons_trained"] = len(results["horizons_trained"])
    
    # 🆕 Total summary
    total_trained = sum(
        len(results["results_per_horizon"].get(h, {}).get("models_trained", []))
        for h in results["horizons_trained"]
    )
    total_failed = sum(
        len(results["results_per_horizon"].get(h, {}).get("models_failed", []))
        for h in results["horizons_trained"]
    )
    results["total_models_trained"] = total_trained
    results["total_models_failed"] = total_failed

    return results


# ==========================================
# CLI TEST
# ==========================================

if __name__ == "__main__":
    print("=" * 70)
    print("ML TRAIN v2 - MEMORY OPTIMIZED")
    print("=" * 70)
    print(f"Available models: {get_available_models()}")
    print(f"  - Random Forest: ✅ (n_estimators=80, max_depth=8)")
    print(f"  - XGBoost: {'✅' if HAS_XGBOOST else '❌'} (n_estimators=100, max_depth=5)")
    print(f"  - LightGBM: {'✅' if HAS_LIGHTGBM else '❌'} (n_estimators=100, max_depth=6)")

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

            print(f"\n✅ Trained {results.get('total_models_trained', 0)} models total")
            if results.get("total_models_failed", 0) > 0:
                print(f"❌ Failed: {results['total_models_failed']} models")
    except Exception as e:
        print(f"❌ {e}")
        import traceback
        traceback.print_exc()
