# recommender/utils.py
"""
Utility functions to load the saved model artifact and perform predictions.
This version is defensive (lazy loads artifact, clear errors) and exposes:
- predict_crop(input_dict) -> (crop_name, confidence)
- predict_crop_topk(input_dict, k=3) -> list[{"crop":..., "confidence":...}]
"""

import os
import joblib
import numpy as np

# Cached objects
_MODEL = None
_LABEL_ENCODER = None
_FEATURES = None
_ARTIFACT_PATH = None
_LOAD_ERROR = None

def _artifact_path():
    """
    Compute the path to the joblib artifact relative to this file.
    Works both in REPL and when run as a script.
    """
    global _ARTIFACT_PATH
    if _ARTIFACT_PATH:
        return _ARTIFACT_PATH

    # base folder = project root (croprec/)
    # file is at croprec/recommender/utils.py -> go up one directory to croprec
    this_dir = os.path.dirname(os.path.abspath(__file__))          # .../croprec/recommender
    base_dir = os.path.dirname(this_dir)                           # .../croprec
    artifact = os.path.join(base_dir, "recommender", "ml", "crop_recommender_v1.joblib")
    _ARTIFACT_PATH = artifact
    return artifact

def _load_artifact():
    """
    Load and cache the artifact dict from joblib. Raises FileNotFoundError
    or RuntimeError if artifact is malformed.
    """
    global _MODEL, _LABEL_ENCODER, _FEATURES, _LOAD_ERROR

    # if previous load errored, re-raise that on attempt
    if _LOAD_ERROR is not None:
        raise _LOAD_ERROR

    if _MODEL is not None and _LABEL_ENCODER is not None and _FEATURES is not None:
        return _MODEL, _LABEL_ENCODER, _FEATURES

    path = _artifact_path()
    if not os.path.exists(path):
        err = FileNotFoundError(f"Model artifact not found at: {path}")
        _LOAD_ERROR = err
        raise err

    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        err = RuntimeError("Loaded artifact must be a dict with keys: model, label_encoder, features")
        _LOAD_ERROR = err
        raise err

    _MODEL = artifact.get("model")
    _LABEL_ENCODER = artifact.get("label_encoder")
    _FEATURES = artifact.get("features")

    if _MODEL is None or _LABEL_ENCODER is None or _FEATURES is None:
        err = RuntimeError("Model artifact missing one of: 'model', 'label_encoder', 'features'")
        _LOAD_ERROR = err
        raise err

    return _MODEL, _LABEL_ENCODER, _FEATURES

def predict_crop(input_dict):
    """
    Predict the best crop and return (crop_name:str, confidence:float 0..1).
    input_dict should contain numeric values for each feature name in artifact 'features'.
    Missing keys default to 0.0.
    """
    model, le, features = _load_artifact()

    # Build feature vector in the expected order
    row = []
    for f in features:
        val = input_dict.get(f, 0)
        try:
            row.append(float(val))
        except Exception as exc:
            raise ValueError(f"Cannot convert feature '{f}' value {val!r} to float: {exc}")

    X = np.array(row).reshape(1, -1)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        pred_idx = int(model.predict(X)[0])
        confidence = float(np.max(probs))
    else:
        pred_idx = int(model.predict(X)[0])
        confidence = 1.0

    try:
        crop_name = le.inverse_transform([pred_idx])[0]
    except Exception:
        # fallback if label encoder not standard
        classes = list(getattr(le, "classes_", []))
        crop_name = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)

    return crop_name, confidence

def predict_crop_topk(input_dict, k=3):
    """
    Return top-k predictions as a list of dicts: [{"crop":name, "confidence":0.81}, ...]
    """
    model, le, features = _load_artifact()

    row = []
    for f in features:
        val = input_dict.get(f, 0)
        try:
            row.append(float(val))
        except Exception as exc:
            raise ValueError(f"Cannot convert feature '{f}' value {val!r} to float: {exc}")

    X = np.array(row).reshape(1, -1)

    if not hasattr(model, "predict_proba"):
        # if model lacks probabilities, return single prediction only
        pred_idx = int(model.predict(X)[0])
        try:
            crop_name = le.inverse_transform([pred_idx])[0]
        except Exception:
            classes = list(getattr(le, "classes_", []))
            crop_name = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
        return [{"crop": crop_name, "confidence": 1.0}]

    probs = model.predict_proba(X)[0]
    topk_idx = np.argsort(probs)[::-1][:k]

    out = []
    for idx in topk_idx:
        try:
            name = le.inverse_transform([int(idx)])[0]
        except Exception:
            classes = list(getattr(le, "classes_", []))
            name = classes[int(idx)] if int(idx) < len(classes) else str(idx)
        out.append({"crop": name, "confidence": float(probs[int(idx)])})

    return out