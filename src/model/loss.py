"""
Loss of the particle-transformer autoencoder: masked reconstruction error + presence BCE.

    w[b, i, f] = mask[b, i] * valid[i, f]          object present AND feature defined for the slot
    e          = reco - target                      phi column: wrapped to [-pi, pi)
    err        = e^2                                btag column: BCEWithLogits(reco, target) instead, the target is 0 / 1
    mse[b]     = sum_{i,f} w err / sum_{i,f} w      mean over the entries the event actually has
    bce[b]     = mean_i BCEWithLogits(presence_logits[b, i], mask[b, i])
    loss       = mean_b ( mse[b] + presence_weight * bce[b] )

Both terms are computed per event first, so with reduction="none" the same function returns the
per-event anomaly score. The MSE is weighted by the TRUE presence of the input, never by the
predicted one, so the presence head is trained only through the BCE term.
"""
import math

import torch
import torch.nn.functional as F

from src.processing.objects import FEATURES

PHI = FEATURES.index("phi")
BINARY = [FEATURES.index("btag")]  # 0 / 1 targets: the model outputs a logit and the error is a BCE, not a squared error


def wrap_phi(delta):
    """Wrap an angle difference to [-pi, pi)."""
    return torch.remainder(delta + math.pi, 2 * math.pi) - math.pi


def compute_loss(reco, presence_logits, target, mask, valid, presence_weight=1.0, reduction="mean"):
    """
    reco, target     (B, 13, 7)  decoder output and the normalized input it should reproduce
    presence_logits  (B, 13)     one logit per slot
    mask             (B, 13)     1 where the object exists in the input
    valid            (13, 7)     which feature is defined for which slot (src.processing.VALID)

    Returns (loss, components). components: "mse" (the reconstruction term; squared error, BCE for btag)
    and "bce" (presence), as batch means or per event (B,) with reduction="none", and "mse_per_feature"
    (7,), the batch mean of each feature's error over its defined entries, to see which features are
    being learned. All detached, for logging.
    """
    valid = torch.as_tensor(valid, device=reco.device, dtype=torch.float32)

    # weight indicates if this value will be used in loss computation
    # only an existing object with a defined feature (non defined feature can be btag for muon) 
    # will be non zero here. So shape is batchsize, num objects, num features
    weight = mask.float()[:, :, None] * valid[None]  # (B, 13, 7)

    reco, target = reco.float(), target.float()
    is_phi = torch.zeros(reco.shape[-1], dtype=torch.bool, device=reco.device)
    is_phi[PHI] = True
    is_binary = torch.zeros_like(is_phi)
    is_binary[BINARY] = True
    error = torch.where(is_phi, wrap_phi(reco - target), reco - target)
    binary_error = F.binary_cross_entropy_with_logits(reco, target.clamp(0.0, 1.0), reduction="none")
    squared = weight * torch.where(is_binary, binary_error, error**2)  # per-entry reconstruction error

    # loss is averaged over num of valid features, not over num of particles, so that the loss is not biased by the number of particles in the event
    mse = squared.sum(dim=(1, 2)) / weight.sum(dim=(1, 2))  # (B,); never 0 / 0: photons and MET are always present
    bce = F.binary_cross_entropy_with_logits(presence_logits.float(), mask.float(), reduction="none").mean(dim=1)  # (B,)
    loss = mse + presence_weight * bce

    components = {"mse": mse, "bce": bce, "mse_per_feature": squared.sum(dim=(0, 1)) / weight.sum(dim=(0, 1)).clamp(min=1)}
    if reduction == "mean":
        loss, components["mse"], components["bce"] = loss.mean(), mse.mean(), bce.mean()
    elif reduction != "none":
        raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")
    return loss, {name: value.detach() for name, value in components.items()}
