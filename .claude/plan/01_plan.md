# Particle-transformer autoencoder: design plan

Status: design discussed and agreed on 2026-09-15, no code written yet. Open decisions are
collected in section 9. Companion documents in this folder:

- `01_plan.md` (this file): the plan.
- `02_design_qa.md`: the design Q&A that led to the plan, with the numerical checks quoted.
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

## 2. Data

Source: the same Delphes `prod_12` skims as the sibling repo (`LOCAL_BASE_DIR` from `setup.sh`,
`version_prod_12_all/SkimEvents/<process>/.../skimmed.h5`), diphoton mass window 105-145 GeV.
Processes: `nonres_yy_jjj`, `ggh_yy`, `vbf_yy`, `vh_yy`, `ttH_yy` (SM, process_id 1..5) and
`WN_HyyN_{150,200,300,600}` (BSM, process_id -1..-4).

### 2.1 Object slots

Slots are fixed, typed, and pt-ordered within a type. This is the central structural fact: the
decoder never has to decide *which* object a slot is, only whether it exists and what its kinematics are.

| slot | object     | physically defined features                  |
|------|------------|----------------------------------------------|
| 0-3  | jet1-4     | pt, eta, phi, m, btag                        |
| 4-5  | fatjet1-2  | pt, eta, phi, m, btag, tau32, tau43 (tau21 also available) |
| 6-7  | electron1-2| pt, eta, phi (m is constant, excluded)       |
| 8-9  | muon1-2    | pt, eta, phi                                 |
| 10   | MET        | pt, phi                                      |
| 11   | diphoton (optional, see 9) | pt, eta, delta_R (mass excluded to avoid sculpting) |

### 2.2 Tensors produced by the data module

- `x`: `(B, M, F)` reconstruction target. Proposed `F = 8`:
  `[pt, eta, cos(phi_rel), sin(phi_rel), m, btag, tau32, tau43]`, normalized (see 2.3).
  If the diphoton becomes slot 11, add one `delta_R` column valid only for that slot (`F = 9`).
- `x_in`: `(B, M, F + n_types)` encoder input = `x` with the type one-hot appended
  (jet, fatjet, electron, muon, met[, diphoton]), one-hot zeroed for absent slots as in production.
- `present`: `(B, M)` float, 1 where the object exists (`pt` finite and `> 0`).
- `valid`: `(M, F)` constant bool, which features are structurally defined per slot (table above).
  The reconstruction loss is computed only where `present[:, i] * valid[i, f]` is 1.
- labels: `process_id`, `group_id`, `sb_id`, `diphoton_mass`, `event_weight`.

### 2.3 Normalization and angles

- Normalize per feature **per object type** (jets, fatjets, leptons, MET), fitted on the training
  split only, NaN-ignoring, log for `pt` and `m`, then NaN -> 0. Reuse the sibling repo's normalizer
  functions. Production ParT fits one mean/std per feature over all object types; per type keeps the
  MSE comparable across slots (fatjet pt is much larger than jet pt).
- Rotate every event so the diphoton system has `phi = 0` (photon phis are in the h5 as
  `sel_photon1_phi`, `sel_photon2_phi`; MET is the fallback reference), then encode each object's
  relative phi as `(cos, sin)`. MSE on raw phi is wrong at +-pi, and the rotation removes an
  irrelevant degree of freedom the latent would otherwise have to carry.
- `eta -> -eta` reflection is an optional augmentation, not for v1.

### 2.4 Sparsity of the slots (measured, first 40k rows per process, 105-145 GeV)

| process        | jet1 | jet2 | jet3 | jet4 | fatjet1 | fatjet2 | el1  | mu1  | mean n_jets |
|----------------|------|------|------|------|---------|---------|------|------|-------------|
| nonres yy+jjj  | 0.46 | 0.18 | 0.06 | 0.02 | 0.01    | 0.00    | 0.00 | 0.00 | 0.7 |
| ttH            | 1.00 | 0.99 | 0.93 | 0.79 | 0.40    | 0.18    | 0.13 | 0.16 | 3.7 |
| WN 150         | 0.87 | 0.63 | 0.35 | 0.17 | 0.26    | 0.12    | 0.08 | 0.09 | 2.0 |
| WN 600         | 0.95 | 0.80 | 0.57 | 0.35 | 0.86    | 0.43    | 0.09 | 0.10 | 2.7 |

MET is present in 100% of events; el2 and mu2 in <1%. Consequences: for most background events the
only object is MET, absence is the majority state for 9 of 11 slots, and the presence pattern alone
separates signal from background. Plain MSE against the zero-filled tensor is therefore the wrong
loss (an absent jet and a real jet at the mean log-pt have identical targets). See section 4.

## 3. Model

### 3.1 Encoder `SlotEncoder`

1. Per-object embedding MLP: `F + n_types -> d` (`d = 64` for the demonstrator, 128 later), GELU.
   Optionally add a learned slot embedding (same shape as the decoder's `Q`) so the encoder knows the
   rank inside a type; the one-hot only gives the type.
2. `L_enc` bidirectional self-attention blocks (`nn.TransformerEncoderLayer`, `batch_first=True`,
   `norm_first=True`), `src_key_padding_mask = present == 0`.
3. Pooling: a learned class token cross-attending to the object tokens (the `ClassAttnDeParT` idea),
   or a masked mean. Then `Linear(d -> 2k)` giving `mu, logvar` (`k = 2..8`). `z = mu + eps*std`
   (VAE) or `z = mu` (deterministic AE), see 9.
4. High-level diphoton features: either a 12th token (recommended, uniform treatment) or
   concatenated after pooling as `DPTModel` does.

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
  src/data/preprocess.py      # sibling loader with the extended column list
  src/data/objects.py         # (x, x_in, present, valid, labels); phi rotation; per-type normalization
  src/model/blocks.py         # embedding MLP, attention block wrapper, adaLN (v2)
  src/model/particle_ae.py    # SlotEncoder, SlotDecoder, ParticleAE
  src/model/losses.py         # reconstruction_loss, SupCon (copy), KL (copy)
  src/train/loop.py           # fit / predict, after the notebook version is stable
  notebooks/particle_ae_demo.ipynb
  tests/                      # shape + symmetry + gradient checks (from .claude/plan/checks)
  .claude/plan/               # this folder
```

1. Copy `setup.sh` and `configs/file_dict.py`. Extend the preprocess column list: jets 1-4
   (`pt, eta, phi, m, btag`), fatjets 1-2 (`+ tau21, tau32, tau43`), electrons/muons 1-2
   (`pt, eta, phi, m`), `met_pt, met_phi`, `sel_photon{1,2}_phi`, `diphoton_pt, diphoton_delta_R,
   diphoton_mass`, weights.
2. `objects.py`: build the tensors. Test on a small slice: shapes, presence fractions reproduce the
   table in 2.4, `valid` mask, rotation puts the diphoton at `phi = 0`, inverse normalization.
3. `particle_ae.py`: encoder and decoder. Test forward shapes and the two symmetry facts
   (`checks/slot_query_check.py`, `checks/cross_attn_check.py`).
4. `losses.py`: reconstruction loss. Test gradient flow (`checks/presence_grad_check.py`).
5. Notebook: load, normalize, train with `lam_con = 0` (pure autoencoder), evaluation plots of
   section 6.
6. Add SupCon (`lam_con > 0`) and KL; compare latent separation vs reconstruction.
7. Baselines and ablations of section 7.

## 9. Open decisions (owner: Runze)

- presence head: independent Bernoulli per slot (v1 proposal) vs count per type;
- diphoton system as a 12th token (proposal) vs high-level concat after pooling;
- latent dimension `k`, and deterministic AE vs VAE reparameterization + KL;
- v1 without the interaction tensor (proposal);
- normalization per object type (proposal) vs one per feature over all objects (production).

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
