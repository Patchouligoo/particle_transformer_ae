# Particle-transformer autoencoder: design plan

Status: model design agreed on 2026-09-15; data-chain decisions finalized later the same day and
recorded in section 2 and in `03_data_chain_implementation.md`. No code written yet. Open decisions
are collected in section 9. Companion documents in this folder:

- `01_plan.md` (this file): the plan.
- `02_design_qa.md`: the design Q&A that led to the plan, with the numerical checks quoted.
- `03_data_chain_implementation.md`: step-by-step implementation plan of the data loading chain,
  the first thing to build.
- `checks/*.py`: the small scripts whose outputs are quoted in the Q&A. Run with the haxad3 python:
  `/global/common/software/m3246/HAXAD/software/haxad3/bin/python checks/<name>.py`

Related repos (read-only reference, do not modify from here):

- `~/Misc/Contrastive_embedding/contrast_embed`: production HAXAD ParT encoder. The object-level
  input format is defined in `src/model/parT/features.py`; attention blocks in
  `src/model/parT/DePartLayer.py`; model in `src/model/parT/dpt_model.py`.
- `~/Misc/Invertible_eocnder/invertible_contrastive_encoder`: sibling demonstrator repo. Reuse its
  `setup.sh`, `configs/file_dict.py`, `src/data/preprocess.py` (chunked h5 loading, event weights,
  105-145 GeV window), `src/data/normalization.py` (NaN-aware fit, invertible), and
  `src/model/contrastive_loss.py` (SupCon, KL). The invertible RealNVP model there is NOT part of
  this project.

## 1. Goal

A demonstrator autoencoder in which both encoder and decoder are transformers over physics objects:

- input: the same object-level tensor as the production ParT, `(B, M objects with padding, N features)`
  plus a presence mask;
- bottleneck: one low-dimensional latent vector `z` per event (VAE-style `mu, logvar` optional);
- output: a tensor of the same shape as the input, trained with a reconstruction loss;
- optional: supervised contrastive loss (SupCon) on `z` plus KL, so the latent is separated by process
  AND decodable back to objects. This is the same motivation as the sibling repo, on object-level
  inputs instead of 6 tabular features.

Intended uses: an anomaly score from reconstruction (presence NLL plus kinematic MSE), an
interpretable latent (decode `z` to objects), and a comparison against the SupCon+KL encoder.

## 2. Data (decided 2026-09-15; implementation in `03_data_chain_implementation.md`)

Source: the same Delphes `prod_12` skims as the sibling repo (`configs/file_dict.py`, `LOCAL_BASE_DIR`
from `setup.sh`), diphoton mass window 105-145 GeV, units MeV. Processes: `nonres_yy_jjj`, `ggh_yy`,
`vbf_yy`, `vh_yy`, `ttH_yy` (process_id 1..5) and `WN_HyyN_{150,200,300,600}` (-1..-4). This project
deliberately allows the diphoton mass and the photon pt: it is a transformer-autoencoder study, not a
HAXAD analysis input, so no anti-sculpting measures are taken.

### 2.1 Slots and features

13 fixed, typed slots, pt-ordered within a type: photon1, photon2, jet1..jet4, fatjet1, fatjet2,
el1, el2, mu1, mu2, met. Seven features in this order: `[pt, eta, phi, m, btag, tau32, tau43]`.
No particle-ID feature: slot identity is fixed, the model learns a slot embedding. Structural
validity (constant, never predicted):

| slot | pt | eta | phi | m | btag | tau32 | tau43 |
|---|---|---|---|---|---|---|---|
| photon1, photon2 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| jet1..jet4 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| fatjet1, fatjet2 | 1 | 1 | 1 | 1 | 0 | 1 | 1 |
| el1, el2, mu1, mu2 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| met | 1 | 0 | 1 | 0 | 0 | 0 | 0 |

### 2.2 Arrays produced by the data chain

| key | shape | content |
|---|---|---|
| `particle_level` | (n, 13, 7) | raw physical values, NaN where absent or undefined; 0 there after normalization |
| `mask` | (n, 13) | 1 where the object exists (pt finite and > 0); the presence head's target |
| `valid` | (13, 7) | the table above |
| `kin_raw` | (n, 13, 4) | pt, eta, rotated phi, m (fatjets only, else 0) for pairwise features and a later per-batch rebuild |
| `interaction` | (n, 13, 13, 6) | dEta, dPhi, dR, dpt_ratio, pt_ij, m_ij from the stored four-vectors; photon pair m_ij is m_yy |
| `phi_yy` | (n,) | diphoton azimuth removed from every phi |
| `process_id`, `event_weight`, `diphoton_mass`, `diphoton_pt`, `diphoton_delta_R` | (n,) | labels and metadata |

### 2.3 Conventions

- **Phi**: every object's phi (photons and MET included) is measured from the massless diphoton
  azimuth and wrapped to [-pi, pi), then stored RAW: no normalization, no cos/sin. Periodicity is
  handled in the loss (section 4). After rotation the pt-weighted sines of the two photons cancel.
- **Masses**: fatjet mass is a substructure feature; all other objects are massless. Pairwise
  four-vectors use the stored masses.
- **Normalization**: pooled per feature over all slots (haxad convention), fitted NaN-ignoring on the
  training sample: log mean-std for pt, m, pt_ij, m_ij; mean-std for dpt_ratio; clip to [0, 1] for
  btag; none for eta, phi, tau, dEta, dPhi, dR. NaN -> 0 afterwards. Invertible, parameters in JSON.
- **Loading**: the first N events inside the mass window per process, one contiguous chunked read
  (contiguous slices were checked to be representative); weights rescaled by rows in file over rows
  consumed, as in the sibling repo.

### 2.4 Measured facts (first 20k-40k rows per process, 105-145 GeV window)

Presence fractions:

| process        | jet1 | jet2 | jet3 | jet4 | fatjet1 | fatjet2 | el1  | mu1  | mean n_jets |
|----------------|------|------|------|------|---------|---------|------|------|-------------|
| nonres yy+jjj  | 0.46 | 0.18 | 0.06 | 0.02 | 0.01    | 0.00    | 0.00 | 0.00 | 0.7 |
| ttH            | 1.00 | 0.99 | 0.93 | 0.79 | 0.40    | 0.18    | 0.13 | 0.16 | 3.7 |
| WN 150         | 0.87 | 0.63 | 0.35 | 0.17 | 0.26    | 0.12    | 0.08 | 0.09 | 2.0 |
| WN 600         | 0.95 | 0.80 | 0.57 | 0.35 | 0.86    | 0.43    | 0.09 | 0.10 | 2.7 |

Photons and MET are present in 100 percent of events; el2 and mu2 in under 1 percent. For most
background events the only objects are the two photons and MET, so the presence pattern alone
separates signal from background, and a plain MSE against the zero-filled tensor is the wrong loss
(section 4). Other facts: missing objects are NaN in the files; photon m = 0, electron m = 0.511 MeV,
muon m = 105.7 MeV, fatjet btag identically 0; median m/pt 0.10-0.12 for jets, 0.28-0.39 for fatjets;
after rotation 69 percent of background leading jets sit within 0.5 rad of phi = +-pi (they recoil
against the diphoton), which is why a plain MSE on the angle is excluded.

## 3. Model

### 3.1 Encoder `SlotEncoder`

1. Per-object embedding MLP: `F = 7 -> d` (`d = 64` for the demonstrator, 128 later), GELU, plus a
   learned slot embedding (13 x d, the encoder counterpart of the decoder's `Q`) that carries object
   type and rank. There is no particle-ID or one-hot feature in the data (decided 2026-09-15).
   IMPLEMENTED 2026-09-17: `DPTModel.slot_embedding` (13, 128), trunc-normal 0.02, added to the tokens
   right after `node_embedding` in `src/model/dpt_model.py` (a copy of HAXAD's encoder without the
   high-level branch). Verified: z was invariant under a slot permutation to 1e-7 before, changes by
   1e-4 after (same order as the event-to-event spread at init); relabelling an electron as a muon
   changes z, untouched events unchanged. Batches come from `GpuBatchLoader` (whole-batch
   `DPTDataSet.get_batch` on the device, 1.8 ms per batch of 32); torch DataLoader was 30-500x slower.
2. `L_enc` bidirectional self-attention blocks (`nn.TransformerEncoderLayer`, `batch_first=True`,
   `norm_first=True`), `src_key_padding_mask = present == 0`.
3. Pooling: a learned class token cross-attending to the object tokens (the `ClassAttnDeParT` idea),
   or a masked mean. Then `Linear(d -> 2k)` giving `mu, logvar` (`k = 2..8`). `z = mu + eps*std`
   (VAE) or `z = mu` (deterministic AE), see 9.
4. The two photons are ordinary slots (decided 2026-09-15), so there is no separate high-level
   branch for now; `diphoton_pt` and `diphoton_delta_R` travel as metadata only.

v1 uses **no** pairwise interaction tensor (`11x11x6`) and no talking-heads attention. v2 can port
`SelfAttnDeParT` + `CNNEmbedding` from the production repo if the encoder turns out to be the
bottleneck. The decoder can never use the interaction tensor (it is a function of the unknown outputs).

### 3.2 Decoder `SlotDecoder`: parallel slot-query decoder (non-autoregressive)

```python
class SlotDecoder(nn.Module):
    def __init__(self, n_slots, n_feat, latent_dim, d=64, n_layers=3, n_heads=8):
        super().__init__()
        self.slot_query = nn.Parameter(torch.randn(1, n_slots, d) * 0.02)   # learned slot identity
        self.from_latent = nn.Linear(latent_dim, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 3 * d, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers)                 # bidirectional, unmasked
        self.feat_head = nn.Linear(d, n_feat)                                # per-slot features
        self.presence_head = nn.Linear(d, 1)                                 # per-slot logit

    def forward(self, z):                                  # z: (B, k)
        h = self.slot_query + self.from_latent(z).unsqueeze(1)   # (B, n_slots, d)
        h = self.blocks(h)                                        # self-attention among slots
        return self.feat_head(h), self.presence_head(h).squeeze(-1)   # (B, M, F), (B, M)
```

Design facts (derived and checked in `02_design_qa.md`):

- `Q` is the only thing that distinguishes slots. Without it all slot outputs are identical
  (permutation equivariance). It is the positional embedding of the decoder, and also a learned
  per-slot prior. Equivalent to MAE mask tokens + positional embeddings, DETR object queries,
  Perceiver IO output queries.
- `Q + W z` is exactly what cross-attention to a single latent token would compute (softmax over one
  key is 1, output is `W_o W_v z'` independent of the query), so a DETR decoder layer collapses to
  self-attention + FFN here.
- Self-attention among slots enforces joint structure (MET balance, pt ordering, monotone presence,
  consistent pairwise masses) with shared weights, so rare slots (jet4) borrow from common ones.
- No attention mask in the decoder: masking by true presence leaks and is unavailable at inference,
  masking by predicted presence is non-differentiable. The presence head decides.
- Optional upgrades: factor `Q = type_emb + rank_emb`; adaptive LayerNorm conditioning on `z` in
  every block (DiT-style) if the decoder gets deeper than ~3 layers.

### 3.3 Why not GPT-style autoregressive decoding

| | autoregressive decoder | slot decoder (chosen) |
|---|---|---|
| order      | one object at a time, causal mask | all slots in parallel, no mask |
| position   | positional embedding              | learned per-slot query |
| length     | EOS token                          | presence logit per slot |
| training   | teacher forcing, likelihood        | MSE + BCE, one pass |
| inference  | sampling loop                      | one forward pass |
| risks      | exposure bias, posterior collapse (decoder ignores z) | fixed slot layout only |

The autoregressive route needs a density head for continuous features (a generative model, a
different project) and is prone to ignoring `z`, which defeats the purpose. The NLP counterpart to
borrow from is non-autoregressive translation, not GPT. If sampling events is ever wanted, a
conditional flow or diffusion decoder on the set is the right upgrade.

### 3.4 Baseline

A flat MLP decoder `z -> M*F + M` with the identical loss is a legitimate architecture for 11 fixed
slots and must be run as the baseline. The sibling repo's `contrastive_vae_mlp.ipynb` is nearly this.

## 4. Loss

```python
def reconstruction_loss(x_hat, p_logit, x, present, valid, lam=1.0):
    # x, x_hat: (B, M, F); p_logit, present: (B, M); valid: (M, F) bool
    w = present.unsqueeze(-1) * valid                          # (B, M, F)
    mse = ((x_hat - x) ** 2 * w).sum() / w.sum().clamp(min=1)  # per valid present entry
    bce = F.binary_cross_entropy_with_logits(p_logit, present) # mean over B*M
    return bce + lam * mse, {"bce": bce.detach(), "mse": mse.detach()}
```

Total: `L = reconstruction + lam_con * SupCon(z or proj(z), group_id, tau) + beta * KL(mu, logvar)`.

- The **true** presence masks the MSE (teacher forcing on presence). No thresholding anywhere in
  training. The presence head learns from BCE only; gating the MSE with the predicted presence either
  kills its gradient (hard cut) or teaches it to call everything absent (soft gate). Checked in
  `checks/presence_grad_check.py`.
- Per-feature loss setting: each of the 7 features gets a loss type and a weight. `phi` is stored as
  a raw rotated angle; v1 gives it weight 0 (masked out), the switch-on is a wrapped squared error
  `(remainder(phi_hat - phi + pi, 2 pi) - pi)^2`, and a cos/sin target is the third option. Never a
  plain MSE on the angle (69 percent of background leading jets sit at the +-pi seam). See Q7 in
  `02_design_qa.md`.
- Sigmoid/Bernoulli per slot, not softmax over slots. Alternative: a count head per type with
  softmax over `0..n_max` and cross-entropy, which guarantees monotone presence (jet3 implies jet2).
- Normalize the MSE by the number of valid present entries, not by `B*M*F`.
- `pos_weight` per slot is optional for rare slots (el2, mu2).
- Log BCE and MSE separately, plus per-slot presence accuracy and per-(slot, feature) MSE.
- Starting weights: `lam = 1`, `lam_con in {0, 0.1, 1}`, `beta ~ 0.01` if VAE. Expect the known
  tension: SupCon collapses within-class variation, reconstruction needs it. `lam_con` is the knob.

Anomaly score at inference (input available, so the true mask is known, no threshold):

```python
presence_nll = F.binary_cross_entropy_with_logits(p_logit, present, reduction="none").sum(-1)  # (B,)
w = present[..., None] * valid
mse = ((x_hat - x) ** 2 * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)                          # (B,)
score = presence_nll + lam * mse
hard_mask = p_logit > 0     # sigmoid > 0.5, only for drawing a reconstructed event
```

## 5. Training

- Pre-batch all tensors on the GPU (as `DPTDataSet` does), `AdamW`, lr `3e-4` with linear warmup
  (5 epochs) + cosine decay, batch 1024-4096, early stopping on validation total loss, patience 10.
- `WeightedRandomSampler` balanced by `group_id` when `lam_con > 0` (production convention);
  plain shuffling for the pure autoencoder runs.
- 60/20/20 train/val/test split with a fixed seed; normalization fitted on train only.
- Run training on a GPU compute node through the interactive QOS, not on the login node.
- Notebook first (style of `contrastive_vae_mlp.ipynb`: repo path bootstrap, `LOCAL_BASE_DIR` from
  `setup.sh`, explicit loop), move the loop into `src/train/loop.py` once stable.

## 6. Evaluation

- Presence: per-slot ROC AUC and accuracy of `p_logit` vs `present`; predicted vs true multiplicity
  per type.
- Kinematics: per-feature histograms original vs reconstructed, present objects only, per process;
  per-(slot, feature) MSE table.
- Physics closure: pairwise masses and MET balance computed from reconstructed four-vectors vs truth.
- Latent: 2D scatter (`k = 2`) or PCA/UMAP colored by process, as in the sibling notebook.
- Anomaly: score distributions background vs each WN mass, ROC per mass, presence-NLL and MSE
  contributions shown separately.
- Baselines: flat MLP autoencoder with the same loss; SupCon+KL encoder alone for latent separation.

## 7. Ablations, ordered by expected value

1. slot-query transformer decoder vs shared per-slot MLP (no attention) vs flat MLP.
2. `lam_con = 0` vs `> 0`: reconstruction vs separation trade-off.
3. presence: independent Bernoulli vs count per type.
4. latent injection: input addition vs adaLN per block.
5. latent dimension `k`.
6. encoder with vs without the interaction tensor (v2).

## 8. Implementation steps and file layout

```
particle_transformer_ae/
  setup.sh                    # LOCAL_BASE_DIR + conda env, copy from the sibling repo
  configs/file_dict.py        # copy from the sibling repo
  src/processing/preprocess.py     # sibling loader with the extended column list (implemented 2026-09-17)
  src/processing/objects.py        # (particle_level, kin_raw, labels) + constants SLOTS/FEATURES/VALID, presence_mask(); pair features per batch from kin_raw; phi rotation; NaN tau -> 0
  src/processing/normalization.py  # pooled per-feature fit / apply / invert, JSON
  # the end-to-end check script lives outside the repo (Claude scratchpad); see 03 for its results
  src/model/blocks.py         # embedding MLP, attention block wrapper, adaLN (v2)
  src/model/particle_ae.py    # SlotEncoder, SlotDecoder, ParticleAE
  src/model/losses.py         # reconstruction_loss, SupCon (copy), KL (copy)
  src/train/loop.py           # fit / predict, after the notebook version is stable
  notebooks/particle_ae_demo.ipynb
  tests/                      # shape + symmetry + gradient checks (from .claude/plan/checks)
  .claude/plan/               # this folder
```

1. `setup.sh` and `configs/file_dict.py` are already copied. Data chain (`preprocess.py`,
   `objects.py`, `normalization.py`, check script): follow `03_data_chain_implementation.md`.
2. Acceptance for the data chain is the check list in section 3 of that document.
3. `particle_ae.py`: encoder and decoder. Test forward shapes and the two symmetry facts
   (`checks/slot_query_check.py`, `checks/cross_attn_check.py`).
4. `losses.py`: reconstruction loss. Test gradient flow (`checks/presence_grad_check.py`).
5. Notebook: load, normalize, train with `lam_con = 0` (pure autoencoder), evaluation plots of
   section 6.
6. Add SupCon (`lam_con > 0`) and KL; compare latent separation vs reconstruction.
7. Baselines and ablations of section 7.

## 9. Open decisions (owner: Runze)

Resolved 2026-09-15 (data chain): two photon slots instead of a diphoton token; pooled normalization
per feature over all slots; no particle-ID feature; fatjet mass kept, all other masses 0; phi rotated
to the diphoton frame, stored raw, handled in the loss; first N events per process; diphoton mass
allowed.

Still open (model side):
- presence head: independent Bernoulli per slot (v1 proposal) vs count per type;
- latent dimension `k`, and deterministic AE vs VAE reparameterization + KL;
- whether the v1 encoder consumes the interaction tensor (the data chain builds it either way);
- phi in the loss: weight 0 (v1) vs wrapped squared error.

## 10. References

- Masked autoencoders, He et al. 2021, arXiv:2111.06377 (mask tokens + positional embeddings,
  MSE on masked patches only).
- DETR, Carion et al. 2020, arXiv:2005.12872 (object queries, "no object" class, regression only on
  matched objects).
- Perceiver IO, Jaegle et al. 2021, arXiv:2107.14795 (output query array).
- Non-autoregressive NMT, Gu et al. 2018, arXiv:1711.02281.
- Sentence VAE / posterior collapse, Bowman et al. 2016, arXiv:1511.06349.
- Diffusion Transformers / adaLN-Zero, Peebles and Xie 2022, arXiv:2212.09748.
- Supervised contrastive learning, Khosla et al. 2020, arXiv:2004.11362.
- HAXAD signal-aware contrastive latent, Li, Nachman, Noll 2026, arXiv:2603.25794.
