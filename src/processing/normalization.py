"""
Pooled per-feature normalization of the arrays of objects.build_arrays.

HAXAD convention: one parameter set per feature, fitted over all slots (or all object pairs) at once
with NaN-ignoring statistics. After the transform NaN (absent object, undefined feature) becomes 0.
Log-normalized features treat x <= 0 as missing. The inverse restores physical units and puts NaN
back where the object is absent (presence_mask) or the feature undefined (VALID).

The pair features are never held in memory for the whole sample: compute_normalization accumulates
their moments chunk by chunk from kin_raw, and build_interaction_norm turns the fitted parameters
into the spec that src.model.utils.build_interaction_torch applies per batch on the device.

    params   = compute_normalization(arrays)            # fit (on the training sample once there is a split)
    arrays_n = apply_normalization(arrays, params)      # particle_level float32, NaN -> 0; kin_raw untouched
    physical = invert_normalization(arrays_n, params)   # back to MeV / radians
    spec     = build_interaction_norm(params)           # [(type, mean, std), ...] for the per-batch builder
"""
import json
import logging

import numpy as np

from .objects import FEATURES, INTERACTION_FEATURES, NUM_SLOTS, VALID, build_interaction_features, presence_mask

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


def _moments(values, norm_type):
    """(count, sum, sum of squares) of the finite values a mean_std / log_mean_std fit uses."""
    if norm_type not in ("mean_std", "log_mean_std"):
        return np.zeros(3)
    values = np.asarray(values, dtype=np.float64)
    values = _log_positive(values) if norm_type == "log_mean_std" else values
    values = values[np.isfinite(values)]
    return np.array([values.size, values.sum(), np.square(values).sum()])


def _params(norm_type, moments, name):
    """Parameter dict of one feature from its accumulated moments."""
    params = {"type": norm_type}
    if norm_type in ("mean_std", "log_mean_std"):
        count, total, total_sq = moments
        mean = total / count if count else 0.0
        std = np.sqrt(max(total_sq / count - mean**2, 0.0)) if count else 0.0
        if std <= 1e-9 * max(1.0, abs(mean)):  # nothing to fit on, or a constant feature
            logger.warning(f"{name}: {int(count)} values, std={std:.3g}; using mean={mean:.3g}, std=1")
            std = 1.0
        params.update(mean=float(mean), std=float(std))
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


def compute_normalization(arrays, chunk=20_000):
    """Fit one parameter set per feature of particle_level and of the pair features.

    The pair features are built from kin_raw `chunk` events at a time and only their moments are kept.
    Returns a JSON-serializable dict {"particle_level": {feature: {"type", "mean", "std"}}, "interaction": {...}}.
    """
    types = NORM_TYPES["particle_level"]
    x = arrays["particle_level"].astype(np.float64)
    particle = {name: _params(types[name], _moments(x[..., f], types[name]), name) for f, name in enumerate(FEATURES)}

    types = NORM_TYPES["interaction"]
    moments = np.zeros((len(INTERACTION_FEATURES), 3))
    for start in range(0, len(arrays["kin_raw"]), chunk):
        interaction = build_interaction_features(arrays["kin_raw"][start : start + chunk])
        for f, name in enumerate(INTERACTION_FEATURES):
            moments[f] += _moments(interaction[..., f], types[name])
    pair = {name: _params(types[name], moments[f], name) for f, name in enumerate(INTERACTION_FEATURES)}
    return {"particle_level": particle, "interaction": pair}


def apply_normalization(arrays, params):
    """New dict with particle_level (and interaction, if present) normalized: float32, NaN / inf -> 0.
    Other entries, kin_raw included, are shared unchanged."""
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


def build_interaction_norm(params):
    """The fitted interaction parameters as the ordered [(type, mean, std), ...] list that
    src.model.utils.build_interaction_torch applies when it builds the pair features per batch."""
    return [
        (p["type"], float(p.get("mean", 0.0)), float(p.get("std", 1.0)))
        for p in (params["interaction"][name] for name in INTERACTION_FEATURES)
    ]


def save_normalization(params, path):
    with open(path, "w") as file:
        json.dump(params, file, indent=2)


def load_normalization(path):
    with open(path) as file:
        return json.load(file)
