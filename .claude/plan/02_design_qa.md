# Design Q&A: particle-transformer autoencoder

Discussion between Runze Li and Claude on 2026-09-15 that led to `01_plan.md`. Questions are
paraphrased from the original; answers are condensed but keep every technical claim and the
numerical checks. Scripts for the checks are in `checks/` and run with
`/global/common/software/m3246/HAXAD/software/haxad3/bin/python checks/<name>.py`.

---

## Q1. Decoding from one latent vector, and the GPT analogy

**Q (Runze):** I want a particle-transformer autoencoder that reads the same input as the contrastive
encoder, `(batch, M objects with padding, N features)`, and outputs the same shape: embedding +
transformer encoder, compression to a latent vector, transformer decoder, MSE between
reconstruction and input. My transformer experience is taggers: self-attention over M objects,
global pooling, output vector or score. Starting from one latent vector, how do we reconstruct the
input with a decoder? How different is it from GPT, where each object is a token and we generate
the first particle, attend, generate the next, and so on? Is there a standard NLP method to borrow?

**A:** The data is not a sequence, it is a fixed set of 11 typed slots: jet1-4 ordered by pt,
fatjet1-2, el1-2, mu1-2, MET (see `features.py` in the production repo). Slot identity is known
before decoding. In NLP terms the positions are given and the vocabulary is continuous, so the right
analogy is the decoder of a masked autoencoder (MAE) or of DETR, not GPT.

Proposed decoder, a parallel slot-query decoder:

1. Learned slot queries `Q` of shape `(11, d)`, acting as positional embedding plus object type.
   Can be factored as type embedding + rank embedding.
2. Inject the latent by broadcast addition `h_i = Q_i + W z`. With one latent vector this is
   mathematically identical to cross-attention to `z` (see Q3).
3. A few bidirectional self-attention layers over the 11 slot tokens. No causal mask, no padding
   mask. It is the encoder block without the interaction bias, because pairwise features are functions
   of the unknown outputs.
4. Two heads per slot: `Linear(d -> 7)` physical features and `Linear(d -> 1)` presence logit.
   The type one-hot is constant per slot and is not reconstructed.

| | GPT-style decoder | slot decoder |
|---|---|---|
| order | one token at a time, causal | all slots at once |
| position | positional embedding | learned per-slot query |
| length | EOS token | presence logit per slot |
| training | teacher forcing, likelihood | MSE + BCE, one pass |
| inference | sampling loop | one forward pass |

Why not autoregressive: (i) with teacher forcing and MSE the model learns the conditional mean given
the *true* previous objects, then at inference consumes its own predictions and errors accumulate;
(ii) doing it properly on continuous features needs a density head (Gaussian mixture), i.e. a
generative model; (iii) a strong autoregressive decoder can ignore `z` entirely (posterior collapse,
Bowman et al. 2016): jet2 pt is predictable from jet1 pt without any latent, and a meaningful `z` is
the whole goal. The NLP work to borrow is non-autoregressive translation (Gu et al. 2018), which
decodes all positions in parallel from a latent for exactly these reasons. If sampling events or
multimodal per-slot conditionals become the goal, a conditional flow or diffusion decoder on the set
is the upgrade, not GPT-style decoding.

Caveat: with 11 fixed slots a flat MLP decoder `z -> 11 x 8` is legitimate, and the sibling
notebook is nearly that. The transformer's advantages are weight sharing across the four jet slots
(rare slots borrow from common ones) and extensibility. Run the MLP as a baseline with the same loss.

## Q1b. Missing objects

**Q (Runze):** We always lay out 4 jets + 2 fatjets + 2 electrons + ..., but many events have no
fatjets, no electrons, no 3rd/4th jet. How do we handle those? Masking parameters? It is somewhat
like the period or line-break token in a language model.

**A:** Measured presence fractions (first 40k rows, 105-145 GeV window):

| process | jet1 | jet2 | jet3 | jet4 | fatjet1 | fatjet2 | el1 | mu1 | mean n_jets |
|---|---|---|---|---|---|---|---|---|---|
| nonres yy+jjj | 0.46 | 0.18 | 0.06 | 0.02 | 0.01 | 0.00 | 0.00 | 0.00 | 0.7 |
| ttH | 1.00 | 0.99 | 0.93 | 0.79 | 0.40 | 0.18 | 0.13 | 0.16 | 3.7 |
| WN 150 | 0.87 | 0.63 | 0.35 | 0.17 | 0.26 | 0.12 | 0.08 | 0.09 | 2.0 |
| WN 600 | 0.95 | 0.80 | 0.57 | 0.35 | 0.86 | 0.43 | 0.09 | 0.10 | 2.7 |

Two consequences. For most background events the only object is MET, so the presence pattern alone
separates signal from background and the autoencoder learns that first. And plain MSE against the
zero-filled tensor is the wrong loss: after normalization an absent object sits at the feature mean,
so an absent jet and a real jet with average log-pt have identical targets; the model learns to
output near zero for rare slots and pulls real objects toward the mean.

Recommended loss (DETR "no object" class + MAE masked loss):

```python
def reconstruction_loss(x_hat, p_logit, x, present, valid, lam=1.0):
    w = present.unsqueeze(-1) * valid                         # (B, 11, 7)
    mse = ((x_hat - x) ** 2 * w).sum() / w.sum().clamp(min=1)
    bce = F.binary_cross_entropy_with_logits(p_logit, present)
    return bce + lam * mse
```

- True presence at train time is fine: presence is predicted by its own head, so using the true
  mask to select which features count is not a leak.
- The fixed `valid` mask encodes structure (jets have no tau, leptons no btag, MET only pt and phi)
  and removes the structural NaNs from the loss.
- Normalize by the count of valid present entries, not by `B*11*7`.
- The anomaly score is the same quantity per event; reporting BCE and MSE separately tells whether an
  event is anomalous in multiplicity or in kinematics.

On the EOS analogy: EOS handles unknown length in sequential generation. The counterpart here is
DETR's no-object class or MAE's mask token, a per-slot binary. Since presence is monotone within a
type (jet3 present implies jet1, jet2 present), the version closest to a "length token" is a count
head per type: softmax over 0..4 jets, 0..2 fatjets, etc. It can never predict jet3 without jet2.
Start with independent Bernoulli per slot; upgrade if inconsistent predictions matter.

Do not mask attention inside the decoder: masking by true presence leaks at train time and is
unavailable at inference; masking by predicted presence is non-differentiable. MAE's decoder attends
to every mask token. Encoder-side masking stays as in production.

Smaller points: phi is periodic, so reconstruct `(cos, sin)` and rotate each event so the diphoton
has `phi = 0` (photon phis are in the h5, MET as fallback); the interaction tensor stays
encoder-only and is not reconstructed; the high-level diphoton features fit naturally as a 12th
always-present slot. Known limitation: pt ordering makes slot assignment discontinuous when two jets
have nearly equal pt; Hungarian matching within a type is the fix if it ever shows in residuals.

---

## Q2. Why does `slot_query` act as a positional embedding?

**Q (Runze):** For `h_i = Q_i + W z`: `z` is `(B, k)`, `W` is `(k, d)`, so `Wz` is `(B, d)`;
`Q` is `(1, M, d)`; `h` is `(B, M, d)`. Is that correct? What does `Q` help us here?

**A:** The shapes are right (`Wz` is unsqueezed to `(B, 1, d)` and broadcast against `Q`).

Without `Q` every slot receives the identical input `Wz`. Self-attention and the shared feed-forward
layers are permutation-equivariant, so 11 identical inputs give 11 identical outputs, and the feature
head predicts the same 7 numbers for jet1, fatjet2 and MET. Checked with a randomly initialized
3-layer decoder (`checks/slot_query_check.py`):

| slot queries | max difference between slot outputs |
|---|---|
| Q set to zero | 0.000 |
| Q random, std 0.02 | 0.067 |

`Q_i` is the only thing in the decoder that tells slot `i` "you are slot i". That is exactly the job
of a positional embedding: the architecture has no notion of index, so position must be injected as
a vector added to the input.

NLP mapping: in BERT/GPT the input at position `i` is token embedding + positional embedding. Here
the "token content" is `Wz` (event-level information, same for every slot) and the "position" is
`Q_i`. This is literally the MAE decoder: every masked patch is one shared learned mask token,
distinguished only by its positional embedding. DETR calls these vectors object queries; Perceiver IO
calls them the output query array. The encoder needs no positional embedding because each slot's own
pt, eta and one-hot type already differ; in the decoder those features are the unknowns.

`Q` is also a learned per-slot prior. Without attention layers the prediction is linear:

```
x_hat_i = A (Q_i + W z) + b  =  (A Q_i + b)  +  A W z
```

The first term is a slot-specific bias learned from data ("MET always present, no eta or mass",
"jet4 rarely present", "fatjet mass large"); the second is the shared event-dependent shift. The
attention layers make the combination nonlinear and let slot `i` condition on the other slot codes
and `z` jointly.

Mathematically, any decoder from one latent vector is a function `z -> (11 x 8)`. A plain MLP uses
separate output weights per slot and no sharing; the slot-query transformer shares all weights and
adds one `d`-dimensional code per slot. Sharing is what lets jet4 (2% of background events) borrow
from jet1.

Practical notes: init std 0.02 (ViT/MAE convention); concatenating `[Q_i ; Wz]` followed by a linear
layer is equivalent to the sum up to a linear map; factoring `Q_i = type_emb + rank_emb` ties the jet
slots together explicitly and mainly helps if the number of objects per type grows.

---

## Q3. Why self-attention among `h`, not attention between each `h` token and `z`?

**Q (Runze):** `h` goes into the transformer blocks and computes attention across itself, still
self-attention. Why not compute attention between each `h` token and `z`?

**A:** Because with `z` as a single token, cross-attention to it is mathematically empty.
Softmax over a single key is always 1.0, so the output is `W_o W_v z'` for every slot and every
query, with zero dependence on `h_i`. Checked with `nn.MultiheadAttention`
(`checks/cross_attn_check.py`):

```
attention weights, all entries         : [1.0]
max |out_i - out_0| across slots       : 0.00e+00
max |out(h) - out(random queries)|     : 0.00e+00
max |out - W_o W_v z' closed form|     : 0.00e+00
```

So "cross-attend to z" is the same operation as "add a linear projection of z to every slot", i.e.
`h_i = Q_i + W z`. A DETR decoder layer is self-attention among queries, cross-attention to memory,
FFN; with one-token memory the middle step collapses into the additive term and what remains is the
proposed block.

What self-attention among slots buys (every `h_i` already contains all of `z`, so it is not about
information access):

- Consistency across slots: MET should balance visible pT in the rotated frame, jets stay
  pt-ordered, presence is monotone within a type, pairwise masses agree. Self-attention lets slot `i`
  read slot `j` and adjust, a message-passing step. DETR's decoder self-attention plays this role and
  is what stops duplicate predictions.
- Event-dependent mixing: the first-layer score `q_i . k_j` is computed from `Q_i + Wz` and
  `Q_j + Wz`, so which slots talk to which changes with the event.
- Honest caveat: the decoder is still a deterministic function of `z`. A shared per-slot MLP with no
  attention is the right ablation; MAE ablations show light decoders often suffice.

Three real options if `z` should enter the attention:

1. Append `z` as a 12th token ("in-context conditioning"). Slot-to-z attention becomes genuine, but
   what flows from `z` is still one fixed vector per head times a scalar weight. The Diffusion
   Transformer paper found in-context and cross-attention conditioning comparable and both worse
   than option 3.
2. Several latent tokens: split `z` into K tokens or expand it with a learned map to `(K, d)`, then
   cross-attend. With a 6-dimensional bottleneck all K tokens are functions of the same 6 numbers, so
   this is a re-parameterization; it becomes meaningful only if the latent is widened into a set of
   tokens, which changes what SupCon acts on.
3. Adaptive LayerNorm: `z` predicts a per-block scale and shift for each LayerNorm, re-injecting `z`
   at every layer instead of once at the input. Current standard for conditioning a transformer on
   one global vector; prevents the input injection from washing out in deep residual stacks.

```python
# adaLN sketch inside one block: z -> (gamma, beta) per LayerNorm
gamma, beta = self.cond(z).chunk(2, dim=-1)          # each (B, d)
h = h + self.attn(self.ln1(h) * (1 + gamma[:, None]) + beta[:, None])
```

For a 2-3 layer decoder, adding `z` at the input is enough. v1: additive injection + self-attention
among slots; ablate against a per-slot MLP without attention and against adaLN if the decoder grows.

---

## Q4. The presence head: softmax? threshold at 0.5? gradient flow?

**Q (Runze):** The presence head is a linear projection giving shape `(batch, n_objects)`, right?
How do we use it to decide masking: softmax, then drop predictions below 0.5? Does this cut allow
gradient flow?

**A:** The head is `Linear(d, 1)` applied to every slot token, so its output is `(B, 11)` logits.

Sigmoid, not softmax. The slots are independent yes/no questions and most events have several
present at once; softmax over the 11 slots would force the probabilities to sum to 1. Softmax
belongs only to the count-per-type variant (over 0..4 jets, 0..2 fatjets, ...).

No cut during training. Presence logits are trained with BCE against the true mask; the feature MSE
is masked with the true mask, not the predicted one. Nothing is thresholded in the forward pass, so
both heads receive gradient. Comparison of the three possible wirings
(`checks/presence_grad_check.py`, random logits, B=256):

| loss variant | grad to feature head | grad to presence head |
|---|---|---|
| MSE gated by hard cut `p > 0.5` | only where gate is open | none |
| MSE gated by `sigmoid(p)` | yes | yes, but 100% of entries positive |
| BCE + MSE masked by true presence | yes | yes |

- Hard cut: a step function, zero gradient. The gate was open for half the slots yet the presence
  head received nothing; it would never learn.
- Soft gate: the derivative of `sigmoid(l) * mse` is `sigmoid'(l) * mse >= 0` for every entry, so
  gradient descent lowers every logit: the cheapest way to shrink the loss is to declare objects
  absent. BCE would fight this, but it is a conflict for no benefit.
- Decoupled loss: the DETR structure, classification on all slots and regression only on the ones
  that exist. If the predicted mask ever had to enter the loss, straight-through estimators or
  Gumbel-sigmoid are the tools; nothing here requires them.

The threshold appears only at inference and only for display. For the anomaly score the input event
is available, so the true mask is known and no threshold is involved:

```python
with torch.no_grad():
    x_hat, p_logit = decoder(z)
    presence_nll = F.binary_cross_entropy_with_logits(p_logit, present, reduction="none").sum(-1)  # (B,)
    w = present[..., None] * valid
    mse = ((x_hat - x) ** 2 * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)                       # (B,)
    score = presence_nll + lam * mse
    hard_mask = p_logit > 0          # sigmoid > 0.5; only for drawing a reconstructed event
```

Features of slots predicted absent are "don't care" outputs and are ignored. When histogramming
reconstructed pt or mass, condition on the true mask. Rare slots (el2, mu2, ~0.5% of events) train
fine with BCE, but a fixed 0.5 threshold may never fire for them in a hard reconstruction; per-slot
thresholds or the count head fix that display problem. It does not affect training or the score.

---

# Data chain Q&A (same day, later)

## Q5. Data loading spec and the particle-ID feature

**Q (Runze):** Load the particle features of selected processes and normalize them like the haxad
repo. Objects: 2 photons, 4 jets, 2 fatjets, 2 electrons, 2 muons, MET; 8 features (p4, btag, tau21,
tau32, particle ID); interaction tensor `(n, N, N, n_int)`; normalization params; process labels.
Photons included this time, with pt rather than pt_rel; global features ignored. Particle ID is not
continuous and is implied by the slot order, so maybe drop it?

**A:** Verified in the h5: `sel_photon1/2` reproduce `diphoton_mass` to 0.02 MeV; missing objects are
NaN, never 0; MeV units; photon m = 0, electron m = 0.511 MeV, muon m = 105.7 MeV (constants);
fatjet btag identically 0; jet btag binary; tau in [0, 1].

Particle ID: agreed to drop it from the data. It is constant per slot, so it carries no information as
a reconstruction target, and a scalar integer imposes a spurious ordinal structure. The encoder still
needs the type because self-attention is permutation-equivariant; it gets it from a learned slot
embedding inside the model (encoder counterpart of the decoder's `Q`), which also encodes rank.

Chain: `preprocess.py` (sibling loader, extended columns) -> `objects.py` (haxad slot table with two
photon slots + constant `valid` mask) -> `normalization.py` (haxad pooled normalizer + inverse).

## Q6. Answers to the six loading questions, with checks

Decisions: diphoton mass leakage is allowed (this is an AE study, not a HAXAD input); tau32 + tau43
(tau21 was a typo); pooled normalization over all slots, no per-type switch; load the first N events
instead of sampling.

**Representativeness of contiguous slices** (20k rows each, before the window cut):

| process | slice | in window | mean diphoton pt [GeV] | mean n_jet | jet1 present | mean jet1 pt [GeV] | fatjet1 present |
|---|---|---|---|---|---|---|---|
| yy+jjj | first / middle / last | 0.845 / 0.845 / 0.838 | 36.2 / 37.0 / 36.7 | 0.745 / 0.757 / 0.755 | 0.470 / 0.475 / 0.475 | 67.1 / 68.0 / 67.5 | 0.013 / 0.013 / 0.014 |
| ttH | first / middle / last | 0.998 / 0.998 / 0.998 | 140.6 / 141.4 / 142.8 | 4.84 / 4.84 / 4.83 | 0.999 / 0.999 / 0.999 | 145.4 / 145.1 / 146.1 | 0.404 / 0.407 / 0.405 |
| WN 600 | first / middle / last | 0.999 / 0.999 / 0.999 | 316.7 / 316.5 / 316.0 | 3.00 / 2.99 / 3.02 | 0.951 / 0.946 / 0.951 | 320.0 / 320.9 / 321.0 | 0.857 / 0.853 / 0.858 |

So first-N loading is statistically fine and a single contiguous read.

**Phi seam after rotating to the diphoton frame.** Fraction of present objects within 0.5 rad of
+-pi (the seam of a single angle):

| object | yy+jjj | ttH | WN 600 |
|---|---|---|---|
| photon1 | 0.00 | 0.00 | 0.00 |
| photon2 | 0.23 | 0.06 | 0.01 |
| jet1 | 0.69 | 0.36 | 0.26 |
| fatjet1 | 0.27 | 0.26 | 0.13 |
| MET | 0.30 | 0.29 | 0.36 |

The leading jet recoils against the diphoton, so a plain MSE on a single angle would put its
discontinuity exactly where jets live. Moving the seam to 0 hits photon1 instead. Conclusion: never
use a plain MSE on phi; see Q7.

**Massless kinematics.** Median m/pt: jets 0.10-0.12, fatjets 0.28-0.39. Massless pairs shift
pair masses by ~1 percent for jets, fine. Fatjet mass is the main boosted-object handle, so it is kept
as a substructure feature; all other objects are massless.

## Q7. Keep phi raw and mask it out of the MSE?

**Q (Runze):** Can we simply keep phi as it is but mask it out from the MSE?

**A:** Yes; the data chain is unaffected (phi stored as one raw angle, no normalization), and it is
what production does on the input side. Cost: the latent is not pushed to encode azimuth, reconstructed
events have no transverse geometry (no pairwise masses, delta R, MET balance from the output), MET
becomes pt-only and photons pt + eta. The one-line alternative that keeps phi is a wrapped squared
error for that column, `(remainder(phi_hat - phi + pi, 2 pi) - pi)^2`; cos/sin target is the third
option. Decision: make the phi treatment a per-feature loss setting (weight 0 / wrapped / cos-sin) on
the model side; store the raw angle in the data.

## Q8. Final mask list and rotation

**Q (Runze):** Keep it simple. Fatjet mass on, all other masses 0, no flags. Masks: btag only jets
1-4, tau only fatjets, mass only fatjets; the presence head masks non-existing objects. OK? Then:
do the rotation after all, since the diphoton p4 is available in preprocessing and gives the baseline
phi for free.

**A:** Yes. That is the constant `valid` (13 x 7) matrix; MET additionally has no eta. Two masks with
different roles: `valid` is structural and never predicted; `mask` (n, 13) is per event and is the
presence head's target. Reconstruction weight per entry is `mask[:, i] * valid[i, f]`. The
normalization fit skips the same entries (NaN), which become 0 afterwards, as in production. Pairwise
features use each object's stored mass (fatjets theirs, others 0). Rotation: phi measured from the
massless diphoton azimuth, wrapped to [-pi, pi), stored raw; `phi_yy` kept per event to undo it.
Implementation plan: `03_data_chain_implementation.md`.
