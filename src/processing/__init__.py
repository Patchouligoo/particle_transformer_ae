"""Data loading chain: h5 skims -> flat DataFrame -> per-object arrays -> pooled normalization."""
from .preprocess import process_delphes_events
from .objects import build_arrays, build_interaction_features, presence_mask, FEATURES, INTERACTION_FEATURES, SLOTS, VALID
from .normalization import (
    compute_normalization,
    apply_normalization,
    invert_normalization,
    save_normalization,
    load_normalization,
)
