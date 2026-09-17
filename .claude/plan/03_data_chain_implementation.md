# Data loading chain: implementation plan

Status: agreed 2026-09-15; IMPLEMENTED 2026-09-17 in `src/processing/` (Runze asked for that path
instead of `src/data/`, so that the notebook stays clean). Deviations from the text below, which is
otherwise what was built:

- the (n, 13, 13, 6) interaction tensor is NOT materialized (Runze, 2026-09-17: compute it per batch
  like HAXAD, memory and time at large n). `build_arrays` returns `kin_raw` (n, 13, 4) = pt, eta, phi, m
  with NaN for absent objects and 0 for the undefined eta of MET / m of non-fatjets;
  `src.model.utils.build_interaction_torch` (HAXAD copy) builds the normalized pair features per batch
  on the device inside `DPTDataSet`; `build_interaction_features(kin_raw)` is its numpy twin (agreement
  4e-4 in normalized units, float32 cancellation in m_ij of light pairs) used for the normalization fit
  and in checks. `compute_normalization` fits the pair-feature parameters by accumulating moments over
  20k-event chunks (identical to a direct nanmean / nanstd to 1e-12); `build_interaction_norm(params)`
  gives the `[(type, mean, std), ...]` spec the torch builder consumes;
- NaN tau32 / tau43 of PRESENT fatjets are stored as 0 (Runze's decision, 2026-09-17), so `mask x valid`
  is the only missingness anywhere. Background: 7-12 percent of present fatjets have NaN tau32 / tau43
  in the skims; 98 percent of them are the diphoton clustered as a fatjet (m ~ 125 GeV, both photons
  within dR < 1). The pattern follows the constituent count exactly: 2 constituents -> tau2 = 0 ->
  tau32 and tau43 NaN; 3 constituents -> tau32 = 0 exactly (already in the file) and tau43 NaN;
  4 constituents -> tau43 = 0 exactly. Storing NaN as 0 therefore continues the file's own convention:
  tau32 = 0 <=> at most 3 constituents, tau43 = 0 <=> at most 4. Rejected alternatives: a per-entry
  `feature_mask` (implemented first, then removed as needless plumbing) and dropping the fatjet when
  tau is NaN (would remove 12 percent of leading fatjets on a technicality, their pt / eta / phi / m
  being fine, and in half of those events fatjet2 would have to move up to keep the slots pt-ordered);
- `valid.sum() == 48` (3+3+16+12+12+2), not 33 as written in an earlier version of this file;
- the output dict has four keys only: `particle_level`, `kin_raw`, `process_id`, `event_weight`
  (Runze, 2026-09-17). Nothing constant or derivable is stored: `valid` became the module constant
  `VALID` (13, 7) next to `SLOTS` / `FEATURES` (it is a property of the feature list, so it belongs there
  or in the loss, not in the data), and the object mask is recomputed on the fly with
  `presence_mask(particle_level)` = `nan_to_num(pt) != 0`, which gives the same answer on raw arrays
  (absent pt = NaN) and on normalized ones (absent pt = 0; verified identical on 3000 events). No
  diphoton_mass, no phi_yy (`diphoton_phi(df)` recomputes the rotation angle), no name lists (use
  `SLOTS`, `FEATURES`, `INTERACTION_FEATURES`). The loss weight is `presence_mask(x)[:, :, None] & VALID`;
- the notebook imports from the package: `from src.processing import process_delphes_events, build_arrays,
  compute_normalization, apply_normalization, invert_normalization`; the cell does
  `combined_df = pd.concat([...]); arrays = build_arrays(combined_df)` and normalizes `arrays`;
- the check script of section 3 was run from Claude's scratchpad, not added to the repo. Results on
  1000 events each of nonres_yy_jjj, ttH_yy, WN_HyyN_600 are in section 5 at the end of this file. This document is specific enough to implement
from directly; decisions are final unless Runze changes them. Scope is the chain

    h5 skims -> flat DataFrame -> per-object arrays (+ interaction tensor) -> pooled normalization

plus a check script. The model, the loss and the training loop are NOT part of this step.

## 0. Decisions this plan implements

| topic | decision |
|---|---|
| objects | 13 fixed slots: photon1, photon2, jet1..jet4, fatjet1, fatjet2, el1, el2, mu1, mu2, met |
| features | `[pt, eta, phi, m, btag, tau32, tau43]`, F = 7; no particle-ID feature |
| masses | fatjet mass kept (substructure feature); every other object massless (m = 0) |
| validity | btag only jets; tau only fatjets; m only fatjets; eta not for MET; see table in 2.2 |
| phi | rotated so the diphoton system has phi = 0, wrapped to [-pi, pi), stored RAW (no normalization, no cos/sin). Periodicity is the loss's job (masked in v1, wrapped squared error later) |
| diphoton mass | allowed. Photon pt (not pt_rel) and the photon-pair m_ij (= m_yy) are used as they are |
| interaction | 6 pairwise features from the stored four-vectors: dEta, dPhi, dR, dpt_ratio, pt_ij, m_ij |
| normalization | pooled per feature over all slots / all pairs (haxad convention), NaN -> 0, invertible |
| loading | FIRST N events inside the 105-145 GeV window per process, contiguous read, no random sampling |
| absent objects | `mask` (n, 13) from pt; the model's presence head is trained against it |

## 1. Inputs

- Files: `configs/file_dict.py` -> `process_dict[name]["file"]`, `["process_id"]` (1..5 SM, -1..-4 BSM).
  `LOCAL_BASE_DIR` comes from `setup.sh`; the notebook bootstrap cell already parses it.
- h5 layout (verified): single key `events`, pandas `table` format, 195 columns, ~21.1M rows for
  nonres_yy_jjj, 1.16M for ttH_yy, 0.58M for WN_HyyN_600. Units MeV and radians. Missing objects are
  NaN in every column of the object (never 0). `sel_photon1/2` are the photons that form
  `diphoton_mass` (recomputed massless mass agrees to 0.02 MeV). Photon m = 0, electron m = 0.511,
  muon m = 105.7, fatjet btag = 0 everywhere.
- Columns to read (55):

```
sel_photon{1,2}_{pt,eta,phi}                       6
sel_jet{1,2,3,4}_{pt,eta,phi,btag}                16
sel_fatjet{1,2}_{pt,eta,phi,m,tau32,tau43}        12
sel_electron{1,2}_{pt,eta,phi}                     6
sel_muon{1,2}_{pt,eta,phi}                         6
met_pt, met_phi                                    2
diphoton_mass, diphoton_pt, diphoton_delta_R       3   (metadata)
event_weight, "pythia_xsec [fb]", pythia_filter_efficiency, sumw_presel   4   (weights)
```

## 2. Module layout and public API

```
src/__init__.py
src/processing/__init__.py
src/processing/preprocess.py        h5 -> flat DataFrame with legacy names, first-N events, event weights
src/processing/objects.py           DataFrame -> dict of arrays (particle_level, mask, valid, kin_raw, interaction, phi_yy, labels)
src/processing/normalization.py     pooled fit / apply / invert on the dict, JSON save/load
scripts/check_data_chain.py   end-to-end run on N events per process + the checks of section 7
```

Notebook usage (replaces the TODO cell in `jupyternotebooks/transformer_AE.ipynb`; the notebook is
edited by Runze, not by Claude):

```python
from src.data.preprocess import process_delphes_events
from src.data.objects import build_arrays
from src.data.normalization import compute_normalization, apply_normalization, invert_normalization

frames = [process_delphes_events(p, num_events_per_process=1000) for p in ["nonres_yy_jjj"]]
combined_df = pd.concat(frames, ignore_index=True)          # concatenate BEFORE building arrays

arrays = build_arrays(combined_df)                          # raw physical values, NaN = absent/undefined
norm_params = compute_normalization(arrays)                 # fit (on the training split once one exists)
arrays_n = apply_normalization(arrays, norm_params)         # NaN -> 0, ready for the model
physical = invert_normalization(arrays_n, norm_params)      # back to MeV / rad, NaN restored
```

### 2.1 `src/processing/preprocess.py`

Derived from the sibling repo's `preprocess.py`; same weight convention, different sampling and columns.

```python
MEASUREMENT_REGION_MIN = 105000.0      # MeV
MEASUREMENT_REGION_MAX = 145000.0
LUMINOSITY = 470.0                     # same constant as the sibling repo's event weight
columns = [...]                        # the 55 columns of section 1

def rename_columns(dataframe) -> DataFrame
    # sel_(photon|jet|fatjet|electron|muon)(\d+)_(feature) -> (photon|jet|fatjet|el|mu)(\d+)_(feature)
    # e.g. sel_electron1_pt -> el1_pt, sel_photon2_phi -> photon2_phi; other columns unchanged

def process_delphes_events(process_name, num_events_per_process=100000) -> DataFrame
    # first num_events_per_process events inside the mass window, legacy names,
    # columns: the renamed object columns + diphoton_mass, diphoton_pt, diphoton_delta_R,
    #          event_weight (rescaled), process_id. dtype float32.
```

Algorithm of the private loader `_load_first_events(file_path, process_id, num_events)`:

1. `chunk_rows = clip(2 * num_events, 10_000, 500_000)`. pandas materializes all 195 columns of a
   chunk before dropping the unrequested ones (~1.6 kB per row), so 10k rows cost ~16 MB and 500k
   rows ~800 MB. For N = 1000 background events one 10k-row read suffices (window efficiency 0.84).
2. For each key of the store (there is one, `events`), for `start` in `range(0, nrows, chunk_rows)`:
   `chunk = store.select(key, start=start, stop=stop, columns=kept)`; `in_window` boolean array;
   `positions = flatnonzero(in_window)` (POSITIONAL, never the DataFrame index); take the first
   `need = num_events - kept` of them; append `chunk.iloc[positions]`; stop when `kept == num_events`.
3. `rows_consumed` = file position of the last kept event + 1 (offset of previous keys + `start` +
   position in chunk + 1). If the file runs out first: warning, `rows_consumed = initial_len`.
4. Weight, identical to the sibling repo except for the last factor:
   `event_weight = LUMINOSITY * xsec_fb * event_weight * filter_eff.replace(+-inf, 1) / sumw_presel * (initial_len / rows_consumed)`.
5. `process_id` column, `astype("float32")`, then `rename_columns`, drop the three raw xsec columns.

Why first-N is acceptable: first / middle / last 20k-row slices of each file agree on all summary
statistics checked (window efficiency, mean diphoton pt, mean n_jet, presence fractions, mean jet1
pt, mean MET) to within 1-2 percent; the files carry no kinematic ordering. Numbers in
`02_design_qa.md`, Q6.

### 2.2 `src/processing/objects.py`

```python
FEATURES = ["pt", "eta", "phi", "m", "btag", "tau32", "tau43"]          # F = 7, this order everywhere
INTERACTION_FEATURES = ["dEta", "dPhi", "dR", "dpt_ratio", "pt_ij", "m_ij"]
SLOTS = [  # (name, type, features physically defined for the slot)
    ("photon1", "photon",   ("pt", "eta", "phi")),
    ("photon2", "photon",   ("pt", "eta", "phi")),
    ("jet1",    "jet",      ("pt", "eta", "phi", "btag")),  ... jet4
    ("fatjet1", "fatjet",   ("pt", "eta", "phi", "m", "tau32", "tau43")),  fatjet2
    ("el1",     "electron", ("pt", "eta", "phi")),  el2
    ("mu1",     "muon",     ("pt", "eta", "phi")),  mu2
    ("met",     "met",      ("pt", "phi")),
]
SLOT_NAMES, SLOT_TYPES, NUM_SLOTS = ..., ..., 13

def valid_mask() -> np.ndarray[bool] (13, 7)
def wrap_phi(phi) -> (phi + pi) % (2 pi) - pi
def diphoton_phi(dataframe) -> (n,)            # atan2(sum pt_g sin phi_g, sum pt_g cos phi_g), massless photons
def build_arrays(dataframe, build_interaction=True, interaction_chunk=20_000) -> dict
def build_interaction_features(kin_raw, chunk=20_000) -> (n, 13, 13, 6) float32
```

Validity matrix produced by `valid_mask()`:

| slot | pt | eta | phi | m | btag | tau32 | tau43 |
|---|---|---|---|---|---|---|---|
| photon1, photon2 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| jet1..jet4 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| fatjet1, fatjet2 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| el1, el2, mu1, mu2 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| met | 1 | 0 | 1 | 0 | 0 | 0 | 0 |

`build_arrays` steps:

1. `phi_yy = diphoton_phi(df)` (float64), shape (n,). Kept in the output to undo the rotation.
2. Per slot `s`: `pt = df[f"{name}_pt"]`; `present = isfinite(pt) & (pt > 0)`; `mask[:, s] = present`.
   `phi_rel = wrap_phi(df[f"{name}_phi"] - phi_yy)` for every slot INCLUDING the photons and MET.
   For each feature in `FEATURES` that is defined for the slot: `particle_level[:, s, f] = where(present, value, NaN)`.
   Undefined features stay NaN. Photons and MET are present in 100 percent of events.
3. `kin_raw[:, s] = (pt, eta, phi_rel, m)` with NaN where absent, `eta = 0` for MET and `m = 0` for
   every non-fatjet slot. This is the four-vector input of the pairwise features.
4. Metadata passthrough as arrays: `process_id` (int64), `event_weight`, `diphoton_mass`,
   `diphoton_pt`, `diphoton_delta_R`; plus the lists `feature_names`, `slot_names`, `slot_types`,
   `interaction_features` so the dict is self-describing.
5. `interaction = build_interaction_features(kin_raw)` when `build_interaction`.

Output dict:

| key | shape / type | content |
|---|---|---|
| `particle_level` | (n, 13, 7) float32 | raw physical values; NaN where absent or undefined |
| `mask` | (n, 13) float32 | 1 where the object exists |
| `valid` | (13, 7) bool | table above |
| `kin_raw` | (n, 13, 4) float32 | pt, eta, phi_rel, m for the pair features; NaN where absent |
| `interaction` | (n, 13, 13, 6) float32 | raw pair features; NaN on the diagonal and for pairs with an absent object |
| `phi_yy` | (n,) float32 | rotation angle removed from every phi |
| `process_id`, `event_weight`, `diphoton_mass`, `diphoton_pt`, `diphoton_delta_R` | (n,) | labels and metadata |
| `feature_names`, `slot_names`, `slot_types`, `interaction_features` | lists | bookkeeping |

`build_interaction_features(kin_raw)`, vectorized over the 13 x 13 pairs with broadcasting
`(c, 13, 1)` against `(c, 1, 13)`, in chunks of `chunk` events computed in float64 and stored as
float32 (peak ~300 MB at 20k events):

```
E   = sqrt(m^2 + pt^2 cosh^2 eta),  px = pt cos phi,  py = pt sin phi,  pz = pt sinh eta
dEta      = eta_i - eta_j
dPhi      = wrap_phi(phi_i - phi_j)                       # rotation invariant
dR        = sqrt(dEta^2 + dPhi^2)
dpt_ratio = (pt_i - pt_j) / (pt_i + pt_j)                 # both > 0 when present, no epsilon needed
pt_ij     = sqrt((px_i + px_j)^2 + (py_i + py_j)^2)
m_ij      = sqrt(max((E_i + E_j)^2 - |p_i + p_j|^2, 0))
diagonal i == j -> NaN;  NaN propagates automatically from absent objects
```

Memory per event: particle_level 364 B, kin_raw 208 B, interaction 4056 B. So 100k events: 36 MB +
21 MB + 406 MB; 1M events: 0.36 + 0.21 + 4.1 GB. `kin_raw` is kept so a later torch port can
rebuild the interaction tensor per batch on the GPU (haxad `build_interaction_torch` pattern) once
the event count makes the dense tensor inconvenient. Not part of this step.

### 2.3 `src/processing/normalization.py`

Array-based, pooled per feature, NaN-ignoring, float64 accumulation; the inverse restores NaN using
`mask` and `valid`.

```python
PARTICLE_NORM_TYPES    = {"pt": "log_mean_std", "eta": "none", "phi": "none", "m": "log_mean_std",
                          "btag": "clip01", "tau32": "none", "tau43": "none"}
INTERACTION_NORM_TYPES = {"dEta": "none", "dPhi": "none", "dR": "none", "dpt_ratio": "mean_std",
                          "pt_ij": "log_mean_std", "m_ij": "log_mean_std"}

def compute_normalization(arrays) -> dict       # fits particle_level (n*13 values per feature) and interaction (n*13*13 per feature)
def apply_normalization(arrays, params) -> dict # new dict; particle_level and interaction transformed, NaN/inf -> 0; other keys shared
def invert_normalization(arrays_norm, params) -> dict   # physical units; NaN where mask*valid == 0 (pairs: mask_i*mask_j, off-diagonal)
def invert_particle_level(x, params, mask=None, valid=None) -> (n, 13, 7)   # for decoder outputs later
def save_normalization(params, path) / load_normalization(path)             # JSON
```

Transform definitions (same as haxad `_apply_norm_value`):

- `log_mean_std`: `(log(x) - mean) / std` with `x <= 0` treated as missing; inverse `exp(y * std + mean)`.
- `mean_std`: `(x - mean) / std`; inverse `y * std + mean`.
- `clip01`: `clip(x, 0, 1)`; inverse identity.
- `none`: identity.
- After every forward transform: `nan_to_num(nan=0, posinf=0, neginf=0)`, cast float32.

Fit safeguards, as in the sibling repo: a non-finite mean becomes 0 and a non-finite or ~zero std
becomes 1, each with a warning. Parameter dict is JSON-serializable and self-describing:

```json
{"particle_level": {"pt": {"type": "log_mean_std", "mean": 11.2, "std": 0.9}, "eta": {"type": "none"}, ...},
 "interaction":    {"dEta": {"type": "none"}, "dpt_ratio": {"type": "mean_std", "mean": 0.0, "std": 0.5}, ...},
 "feature_names": ["pt", "eta", "phi", "m", "btag", "tau32", "tau43"],
 "interaction_features": ["dEta", "dPhi", "dR", "dpt_ratio", "pt_ij", "m_ij"]}
```

Fitting on the combined loaded sample is fine for the notebook demonstrator; the real runs fit on the
training split only, once the split exists (training-loop step, not here).

## 3. `scripts/check_data_chain.py`

Command line: `--processes nonres_yy_jjj ttH_yy WN_HyyN_600 --num-events 1000`. It runs the chain and
prints, in this order, failing loudly (assert) on the hard checks:

1. Load time and DataFrame shape; exactly `num_events` rows per process, all inside the window.
2. Shapes: `particle_level (n, 13, 7)`, `mask (n, 13)`, `valid (13, 7)`, `kin_raw (n, 13, 4)`,
   `interaction (n, 13, 13, 6)`, `phi_yy (n,)`. `valid.sum() == 48`.
3. Presence fraction per slot and process. Expected on 1000 events (from the 40k-row measurement,
   1000-event statistical error ~0.015): background jet1 0.46, jet2 0.18, jet3 0.06, fatjet1 0.01,
   leptons 0.00; ttH jet1 1.00, jet4 0.79, fatjet1 0.40, el1 0.13, mu1 0.16; photons and MET 1.00.
4. Rotation: `|sum_g pt_g sin(phi_rel_g)| / sum_g pt_g < 1e-5` for every event (the diphoton sits at
   phi = 0), and `phi_yy` matches `atan2` of the unrotated photon momenta.
5. Photon-pair `m_ij` equals `diphoton_mass`: median absolute difference below 1 MeV.
6. Normalized arrays contain no NaN or inf; every entry with `mask * valid == 0` is exactly 0; the
   pooled `pt` over present-valid entries has mean within 0.05 of 0 and std within 0.05 of 1.
7. Round trip `invert(apply(x))`: max relative deviation below 1e-5 on present-valid entries, NaN
   pattern identical to the raw arrays.
8. Summary: process ids present, sum of event weights per process, `phi` range in [-pi, pi).

Run from the repo root with the haxad3 python:
`LOCAL_BASE_DIR=... /global/common/software/m3246/HAXAD/software/haxad3/bin/python scripts/check_data_chain.py`.
A 1000-event run per process is small enough for the login node; anything at the 100k scale goes
to an interactive compute node.

## 4. Implementation order and acceptance

| step | deliverable | done when |
|---|---|---|
| 1 | `src/__init__.py`, `src/processing/__init__.py` | imports work from the notebook bootstrap |
| 2 | `preprocess.py` | 1000 background events load in a few seconds; rows all in window; renamed columns present; weights finite |
| 3 | `objects.py` | checks 2-5 pass on background + ttH |
| 4 | `normalization.py` | checks 6-7 pass; params round-trip through JSON |
| 5 | `check_data_chain.py` | full report for nonres_yy_jjj, ttH_yy, WN_HyyN_600 |
| 6 | plan folder | `01_plan.md` and `02_design_qa.md` reflect the final decisions (done together with this file) |

Not in this step, deferred to the model/training work: caching arrays to `$SCRATCH` (npz/h5) for
large N, train/val/test split with fit-on-train, the per-batch torch interaction builder, the phi
loss treatment, the encoder slot embedding.

## 5. Check results (2026-09-17; 1000 events per process; login node; haxad3 python; scratchpad script)

| check | result |
|---|---|
| load | 0.1-0.3 s per process; 1000 rows each, all inside the window; sum of weights: nonres 9.80e5, ttH 109.5, WN600 4.6e-5 |
| shapes | particle_level (3000, 13, 7) float32, kin_raw (3000, 13, 4), process_id (3000,) int64, event_weight (3000,); VALID (13, 7) sums to 48; presence_mask (3000, 13) identical before and after normalization; `DPTDataSet.get_batch` returns the (B, 13, 13, 6) pair features built on the fly, equal to the numpy ones to 4e-4 |
| presence | nonres: jet1 .46 jet2 .19 jet3 .06 jet4 .03 fatjet1 .01, leptons .00; ttH: jet1 1.00 jet4 .78 fatjet1 .39 el1 .14 mu1 .17; WN600: jet1 .95 fatjet1 .84 fatjet2 .42; photons and MET 1.00 |
| NaN tau of present fatjets | 327 entries in the h5 with mask x valid = 1 but NaN, all tau32 / tau43, stored as 0; fatjet1: nonres 1 of 14, ttH 16 (tau32) / 26 (tau43) of 386, WN600 69 / 126 of 844. 98 percent of them are fatjets within dR < 1 of both photons, median m 125.0 GeV (the diphoton clustered as a fatjet; 36 percent of all present fatjet1 are such objects). After storing them as 0, every present-valid entry is finite |
| rotation | max of abs(sum pt sin phi_rel) / sum pt over the photons = 5.7e-8; adding phi_yy back reproduces the detector phi of every slot; phi in [-3.1415, 3.1415] |
| pair features | median abs(m_ij(y1, y2) - diphoton_mass) = 0.023 MeV (max 4 MeV); NaN exactly on the diagonal and for pairs with an absent object; dEta and dPhi antisymmetric, m_ij symmetric; dR(y1, jet1) equals the detector-frame value; dEta(y1, met) = eta(y1) |
| normalization | no NaN / inf; zeros exactly where presence_mask x VALID = 0 (pairs: where pair presence = 0); pooled pt over present objects: mean 0.000, std 1.000; fitted pt log-mean 11.40, log-std 1.04 (MeV) |
| round trip | invert(apply(x)): max relative deviation 2.6e-7 (particle_level) and 2.7e-7 (interaction) on non-zero entries; NaN pattern identical to the raw arrays; exact zeros of none / clip01 features are identity; a zero of a log feature is treated as missing by convention (120 such pair entries); JSON save / load identical |
| notebook | `jupyternotebooks/transformer_AE.ipynb` cells 1-2 run as written in 5.7 s for 1000 nonres events |
