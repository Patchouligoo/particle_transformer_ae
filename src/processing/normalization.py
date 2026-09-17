"""
Pooled per-feature normalization of the arrays of objects.build_arrays.

HAXAD convention: one parameter set per feature, fitted over all slots (or all object pairs) at once
with NaN-ignoring statistics. After the transform NaN (absent object, undefined feature) becomes 0.
Log-normalized features treat x <= 0 as missing. The inverse restores physical units and puts NaN
back where the object is absent (presence_mask) or the feature undefined (VALID); for pairs, where
an object is absent or on the diagonal.

    params   = compute_normalization(arrays)            # fit (on the training sample once there is a split)
    arrays_n = apply_normalization(arrays, params)      # float32, NaN -> 0, ready for the model
    physical = invert_normalization(arrays_n, params)   # back to MeV / radians
"""
import json
import logging

import numpy as np

from .objects import FEATURES, INTERACTION_FEATURES, NUM_SLOTS, VALID, presence_mask

logger = logging.getLogger(__name__)

# normalization type per feature; "none" passes the feature through unchanged
NORM_TYPES = {
    "particle_level": {
        "pt": "log_mean_std", "eta": "none", "phi": "none", "m": "log_mean_std",
        "btag": "clip01", "tau32": "none", "tau43": "none",
    },
    "interaction": {
        "dEta": "none", "dPhi": "none", "dR": "none", "dpt_ratio": "mean_std",
        "pt_ij": "log_mean_std", "m_ij": "log_mean_std",
    },
}
FEATURE_NAMES = {"particle_level": FEATURES, "interaction": INTERACTION_FEATURES}


def _log_positive(x):
    return np.log(np.where(x > 0, x, np.nan))


def _fit(x, norm_type, name):
    """Parameters of one feature from all its finite values."""
    params = {"type": norm_type}
    if norm_type in ("mean_std", "log_mean_std"):
        values = _log_positive(x) if norm_type == "log_mean_std" else x
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std()) if values.size else 0.0
        if std <= 1e-9 * max(1.0, abs(mean)):  # nothing to fit on, or a constant feature
            logger.warning(f"{name}: {values.size} values, std={std:.3g}; using mean={mean:.3g}, std=1")
            std = 1.0
        params.update(mean=mean, std=std)
    return params


def _forward(x, params):
    kind = params["type"]
    if kind == "log_mean_std":
        return (_log_positive(x) - params["mean"]) / params["std"]
    if kind == "mean_std":
        return (x - params["mean"]) / params["std"]
    if kind == "clip01":
        return np.clip(x, 0.0, 1.0)
    return x


def _backward(y, params):
    kind = params["type"]
    if kind == "log_mean_std":
        return np.exp(y * params["std"] + params["mean"])
    if kind == "mean_std":
        return y * params["std"] + params["mean"]
    return y


def compute_normalization(arrays):
    """Fit one parameter set per feature of particle_level and of interaction (if present).

    Returns a JSON-serializable dict {array key: {feature: {"type", "mean", "std"}}}.
    """
    params = {}
    for key, types in NORM_TYPES.items():
        if key in arrays:
            x = arrays[key].astype(np.float64)
            params[key] = {name: _fit(x[..., f], types[name], name) for f, name in enumerate(FEATURE_NAMES[key])}
    return params


def apply_normalization(arrays, params):
    """New dict with particle_level and interaction normalized (float32, NaN / inf -> 0); other entries shared."""
    out = dict(arrays)
    for key, feature_params in params.items():
        if key in arrays:
            x = arrays[key].astype(np.float64)
            y = np.stack([_forward(x[..., f], feature_params[name]) for f, name in enumerate(FEATURE_NAMES[key])], axis=-1)
            out[key] = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out


def invert_normalization(arrays_norm, params):
    """Undo apply_normalization: physical units, NaN where the object is absent or the feature undefined."""
    present = presence_mask(arrays_norm["particle_level"])
    defined = {
        "particle_level": present[:, :, None] & VALID[None, :, :],
        "interaction": present[:, :, None] & present[:, None, :] & ~np.eye(NUM_SLOTS, dtype=bool),
    }
    out = dict(arrays_norm)
    for key, feature_params in params.items():
        if key in arrays_norm:
            y = arrays_norm[key].astype(np.float64)
            x = np.stack([_backward(y[..., f], feature_params[name]) for f, name in enumerate(FEATURE_NAMES[key])], axis=-1)
            x[~defined[key]] = np.nan
            out[key] = x.astype(np.float32)
    return out


def save_normalization(params, path):
    with open(path, "w") as file:
        json.dump(params, file, indent=2)


def load_normalization(path):
    with open(path) as file:
        return json.load(file)
