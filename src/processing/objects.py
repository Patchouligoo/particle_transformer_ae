"""
Flat DataFrame -> per-object arrays for the transformer.

Every event is described by 13 fixed, pt-ordered object slots with 7 features each:

    slots    : photon1 photon2 jet1 jet2 jet3 jet4 fatjet1 fatjet2 el1 el2 mu1 mu2 met
    features : pt eta phi m btag tau32 tau43

Not every feature is defined for every slot (btag only for jets, m / tau only for fatjets, no eta for
MET): the constant VALID (13, 7) says which ones are. An object that does not exist in an event is NaN
everywhere (0 once normalized), so presence_mask() recovers the objects of an event from pt alone and
nothing has to be stored for it. The one exception in the skims, tau32 / tau43 of a two-constituent
fatjet (tau2 = 0, mostly the diphoton clustered as a fatjet), is stored as 0, the value a
three-constituent fatjet has. Values stay in physical units (MeV, radians); every phi is measured from
the diphoton system (whose azimuth is then 0) and wrapped to [-pi, pi).
"""
import numpy as np

FEATURES = ["pt", "eta", "phi", "m", "btag", "tau32", "tau43"]
INTERACTION_FEATURES = ["dEta", "dPhi", "dR", "dpt_ratio", "pt_ij", "m_ij"]

SLOTS = {  # slot -> features that are physically defined for it
    "photon1": ("pt", "eta", "phi"),
    "photon2": ("pt", "eta", "phi"),
    "jet1": ("pt", "eta", "phi", "btag"),
    "jet2": ("pt", "eta", "phi", "btag"),
    "jet3": ("pt", "eta", "phi", "btag"),
    "jet4": ("pt", "eta", "phi", "btag"),
    "fatjet1": ("pt", "eta", "phi", "m", "tau32", "tau43"),
    "fatjet2": ("pt", "eta", "phi", "m", "tau32", "tau43"),
    "el1": ("pt", "eta", "phi"),
    "el2": ("pt", "eta", "phi"),
    "mu1": ("pt", "eta", "phi"),
    "mu2": ("pt", "eta", "phi"),
    "met": ("pt", "phi"),
}
NUM_SLOTS, NUM_FEATURES = len(SLOTS), len(FEATURES)
VALID = np.array([[f in features for f in FEATURES] for features in SLOTS.values()])  # (13, 7) bool


def presence_mask(particle_level):
    """(n, 13) bool, True where the object exists. Works on raw arrays (absent: pt is NaN) and on
    normalized ones (absent: pt is 0; a present object never has pt exactly 0)."""
    return np.nan_to_num(particle_level[..., FEATURES.index("pt")]) != 0


def wrap_phi(phi):
    """Wrap an angle to [-pi, pi)."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def diphoton_phi(dataframe):
    """Azimuth of the diphoton transverse momentum: the direction every phi is measured from."""
    pt1, pt2 = dataframe["photon1_pt"].to_numpy(np.float64), dataframe["photon2_pt"].to_numpy(np.float64)
    phi1, phi2 = dataframe["photon1_phi"].to_numpy(np.float64), dataframe["photon2_phi"].to_numpy(np.float64)
    return np.arctan2(pt1 * np.sin(phi1) + pt2 * np.sin(phi2), pt1 * np.cos(phi1) + pt2 * np.cos(phi2))


def build_arrays(dataframe, build_interaction=True):
    """Turn the flat DataFrame of preprocess.process_delphes_events into arrays.

    Returns a dict with
        particle_level  (n, 13, 7)      float32  physical values; NaN where the object is absent or the feature undefined
        interaction     (n, 13, 13, 6)  float32  pair features; NaN on the diagonal and for pairs with an absent object
        process_id      (n,)            int64    label from configs/file_dict.py
        event_weight    (n,)            float32
    Slot / feature order: SLOTS, FEATURES, INTERACTION_FEATURES. presence_mask() and VALID say which
    entries carry a value.
    """
    n = len(dataframe)
    phi_yy = diphoton_phi(dataframe)
    particle_level = np.full((n, NUM_SLOTS, NUM_FEATURES), np.nan, dtype=np.float32)
    for s, (slot, features) in enumerate(SLOTS.items()):
        pt = dataframe[f"{slot}_pt"].to_numpy(np.float64)
        present = np.isfinite(pt) & (pt > 0)  # an absent object is NaN in every column
        for feature in features:
            values = dataframe[f"{slot}_{feature}"].to_numpy(np.float64)
            if feature == "phi":
                values = wrap_phi(values - phi_yy)
            # nan_to_num: a present fatjet can have NaN tau32 / tau43 (tau2 = 0); stored as 0, see module docstring
            particle_level[present, s, FEATURES.index(feature)] = np.nan_to_num(values[present])

    arrays = {
        "particle_level": particle_level,
        "process_id": dataframe["process_id"].to_numpy().astype(np.int64),
        "event_weight": dataframe["event_weight"].to_numpy(np.float32),
    }
    if build_interaction:
        arrays["interaction"] = build_interaction_features(particle_level)
    return arrays


def build_interaction_features(particle_level, chunk=10_000):
    """(n, 13, 13, 6) pair features from the physical particle_level array.

    Uses its first four features pt, eta, phi, m as the four-vector; an undefined eta or m counts as 0
    (MET at eta = 0, non-fatjets massless). NaN on the diagonal and wherever one object is absent.
    """
    n = len(particle_level)
    interaction = np.empty((n, NUM_SLOTS, NUM_SLOTS, len(INTERACTION_FEATURES)), dtype=np.float32)
    for start in range(0, n, chunk):  # chunked: the intermediates are (chunk, 13, 13) float64
        four_vectors = np.nan_to_num(particle_level[start : start + chunk, :, :4].astype(np.float64))
        interaction[start : start + chunk] = _pair_features(four_vectors)
    present = presence_mask(particle_level)
    pair_present = present[:, :, None] & present[:, None, :] & ~np.eye(NUM_SLOTS, dtype=bool)
    interaction[~pair_present] = np.nan
    return interaction


def _pair_features(kin):
    """kin (c, 13, 4) = pt, eta, phi, m  ->  (c, 13, 13, 6) in INTERACTION_FEATURES order."""
    pt, eta, phi, m = (kin[..., k] for k in range(4))
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    energy = np.sqrt(m**2 + px**2 + py**2 + pz**2)

    def i(a):  # object i along axis 1 ...
        return a[:, :, None]

    def j(a):  # ... broadcast against object j along axis 2
        return a[:, None, :]

    d_eta = i(eta) - j(eta)
    d_phi = wrap_phi(i(phi) - j(phi))
    px_ij, py_ij, pz_ij, e_ij = (i(a) + j(a) for a in (px, py, pz, energy))
    with np.errstate(invalid="ignore", divide="ignore"):  # 0 / 0 for two absent objects, masked by the caller
        dpt_ratio = (i(pt) - j(pt)) / (i(pt) + j(pt))
    features = {
        "dEta": d_eta,
        "dPhi": d_phi,
        "dR": np.hypot(d_eta, d_phi),
        "dpt_ratio": dpt_ratio,
        "pt_ij": np.hypot(px_ij, py_ij),
        "m_ij": np.sqrt(np.maximum(e_ij**2 - px_ij**2 - py_ij**2 - pz_ij**2, 0.0)),
    }
    return np.stack([features[name] for name in INTERACTION_FEATURES], axis=-1)
