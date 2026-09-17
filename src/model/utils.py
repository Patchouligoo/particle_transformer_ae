import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _p4_from_ptephim_torch(p4):
    """Torch port of ``p4_from_ptephim``; p4[..., :] = (pt, eta, phi, m)."""
    pt = p4[..., 0]
    eta = p4[..., 1]
    phi = p4[..., 2]
    m = p4[..., 3]

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)
    E = torch.sqrt(m * m + (pt * torch.cosh(eta)) ** 2)
    return torch.stack((E, px, py, pz), dim=-1)


def _apply_interaction_norm_torch(interaction, interaction_norm):
    """Per-batch equivalent of ``features._apply_norm_value`` for interaction.

    ``interaction``: (..., NUM_INTERACTION_FEATURES). ``interaction_norm`` is the
    ordered list of ``(type, mean, std)`` from ``features.build_interaction_norm``.
    Matches the numpy path bit-for-bit: standardize then nan_to_num.
    """
    channels = []
    for f, (ntype, mean, std) in enumerate(interaction_norm):
        v = interaction[..., f]
        if ntype == "log_mean_std":
            logged = torch.log(torch.where(v > 0, v, torch.full_like(v, float("nan"))))
            out = (logged - mean) / std
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        elif ntype == "mean_std":
            out = (v - mean) / std
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        elif ntype == "none":
            out = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            raise ValueError(f"Unknown interaction normalization type: {ntype}")
        channels.append(out)
    return torch.stack(channels, dim=-1)


def build_interaction_torch(kin_raw, interaction_norm):
    """Build the normalized (B, N, N, 6) interaction tensor from raw kinematics.

    ``kin_raw``: (B, N, 4) float32 raw (pt, eta, phi, m), NaN for missing
    particles. Reproduces ``features.build_arrays`` interaction features +
    ``apply_interaction_norm`` exactly, computed on-device per batch so the dense
    interaction array is never stored for all events.

    Runs in float32 with autocast disabled to guarantee numerical equivalence
    regardless of any surrounding ``torch.autocast`` region.
    """
    device = kin_raw.device
    autocast_ctx = (
        torch.autocast(device_type=device.type, enabled=False)
        if device.type in ("cuda", "cpu")
        else _nullcontext()
    )
    with autocast_ctx:
        kin_raw = kin_raw.float()
        B, N, _ = kin_raw.shape
        pi = kin_raw[:, :, None, :]  # (B, N, 1, 4)
        pj = kin_raw[:, None, :, :]  # (B, 1, N, 4)

        dEta = pi[..., 1] - pj[..., 1]  # broadcast -> (B, N, N)
        dPhi = pi[..., 2] - pj[..., 2]
        dPhi = (dPhi + math.pi) % (2 * math.pi) - math.pi
        dR = torch.sqrt(dEta ** 2 + dPhi ** 2)

        pt_i = pi[..., 0]
        pt_j = pj[..., 0]
        dpt_ratio = (pt_i - pt_j) / (pt_i + pt_j + 1e-8)  # (B, N, N)

        p4_ij = _p4_from_ptephim_torch(pi) + _p4_from_ptephim_torch(pj)  # (B, N, N, 4)
        pt_ij = torch.sqrt(p4_ij[..., 1] ** 2 + p4_ij[..., 2] ** 2)
        m2 = p4_ij[..., 0] ** 2 - (
            p4_ij[..., 1] ** 2 + p4_ij[..., 2] ** 2 + p4_ij[..., 3] ** 2
        )
        m2 = torch.where(m2 < 0, torch.zeros_like(m2), m2)
        m_ij = torch.sqrt(m2)

        interaction = torch.stack([dEta, dPhi, dR, dpt_ratio, pt_ij, m_ij], dim=-1)

        # The numpy builder skips i == j (``continue``), leaving the diagonal at
        # its NaN init -> normalized to 0. Force it to NaN here to match.
        eye = torch.eye(N, dtype=torch.bool, device=device)
        interaction = interaction.masked_fill(eye[None, :, :, None], float("nan"))

        return _apply_interaction_norm_torch(interaction, interaction_norm)


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def p4_from_ptephim(p4_ptetaphim):
    pt = p4_ptetaphim[:, 0]
    eta = p4_ptetaphim[:, 1]
    phi = p4_ptetaphim[:, 2]
    m = p4_ptetaphim[:, 3]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(m * m + (pt * np.cosh(eta)) ** 2)

    return np.stack((E, px, py, pz), axis=-1)


def sum_p4(p1, p2):
    return p1 + p2


def pt_of(p4):
    px = p4[..., 1]
    py = p4[..., 2]
    return np.sqrt(px**2 + py**2)


def invariant_mass(p4):
    E = p4[..., 0]
    px = p4[..., 1]
    py = p4[..., 2]
    pz = p4[..., 3]
    m2 = E**2 - (px**2 + py**2 + pz**2)
    m2 = np.where(m2 < 0, 0, m2)
    return np.sqrt(m2)


def make_norm(normalized_shape, use_rmsnorm=False):
    """Norm factory: RMSNorm (HyperScale-style, no centering/bias) or LayerNorm."""
    if use_rmsnorm:
        return nn.RMSNorm(normalized_shape)
    return nn.LayerNorm(normalized_shape)


class FFN(nn.Module):
    """Transformer FFN with GLU activation (ParT variant)."""

    def __init__(self, dim, expansion, dropout=None, use_rmsnorm=False):
        super().__init__()
        self.dim = dim
        self.expansion = expansion
        self.dropout = dropout

        hidden_dim = int(dim * expansion * 2 / 3)
        self.wide_dense = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_dense = nn.Linear(dim, hidden_dim, bias=False)

        self.ln = make_norm(hidden_dim, use_rmsnorm)
        self.dense = nn.Linear(hidden_dim, dim, bias=False)
        self.layer_dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x):
        value = F.gelu(self.wide_dense(x))
        gate = self.gate_dense(x)
        output = value * gate
        output = self.ln(output)
        output = self.dense(output)
        if self.layer_dropout is not None:
            output = self.layer_dropout(output)
        return output


class LayerScale(nn.Module):
    def __init__(self, ini_value, dim):
        super().__init__()
        self.scale = nn.Parameter(ini_value * torch.ones(dim))

    def forward(self, x):
        return x * self.scale


class FCEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_embeding_layers):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeding_layers = num_embeding_layers
        self.input_dim = input_dim

        layers = [nn.Linear(self.input_dim, embedding_dim), nn.GELU()]
        for _ in range(1, num_embeding_layers):
            layers.append(nn.Linear(embedding_dim, embedding_dim))
            layers.append(nn.GELU())
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class ChannelLayerNorm1d(nn.Module):
    """LayerNorm over the channel dim of a ``(B, C, L)`` tensor.

    Per-position (per pair) normalization that, unlike ``BatchNorm1d``, keeps no
    running statistics — so the many zero-padded interaction pairs do not pollute
    the stats and there is no train/eval running-stat mismatch.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x):  # x: (B, C, L) -> normalize over C
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


def _make_cnn_norm(norm_type, num_channels):
    """Norm factory for the interaction CNN: LayerNorm (padding-agnostic, no
    running stats) or the legacy BatchNorm1d."""
    if norm_type == "layernorm":
        return ChannelLayerNorm1d(num_channels)
    return nn.BatchNorm1d(num_channels)


class CNNEmbedding(nn.Module):
    """Embed pairwise interaction matrix via 1D conv over upper-triangular pairs.

    The upper-triangular layout is fixed (depends only on ``num_particles``), so
    the gather/scatter indices are precomputed once as a static buffer. This
    avoids the data-dependent ``torch.nonzero`` + advanced-index scatter that
    forces a graph break under ``torch.compile``. Numerically identical to the
    previous boolean-mask implementation.
    """

    def __init__(self, num_layers, layer_size, in_dim, out_dim, num_particles,
                 norm_type="batchnorm"):
        super().__init__()
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num_particles = num_particles

        self.conv_layers = [nn.Conv1d(in_dim, layer_size, kernel_size=1)]
        for _ in range(1, num_layers):
            self.conv_layers.append(nn.Conv1d(layer_size, layer_size, kernel_size=1))
        self.conv_layers.append(nn.Conv1d(layer_size, out_dim, kernel_size=1))
        self.conv_layers = nn.ModuleList(self.conv_layers)

        self.norm_layers = [_make_cnn_norm(norm_type, layer_size) for _ in range(num_layers)]
        self.norm_layers.append(_make_cnn_norm(norm_type, out_dim))
        self.norm_layers = nn.ModuleList(self.norm_layers)

        self.activation = nn.GELU()

        # Flat indices of the strict upper triangle in a row-major (N*N) grid.
        # Computed once in eager mode; registered as a buffer so it tracks
        # device/dtype and compiles as a static-shape index_select/index_copy.
        upper = torch.triu(torch.ones(num_particles, num_particles), diagonal=1).bool()
        ul_flat_idx = torch.nonzero(upper.reshape(-1), as_tuple=False).squeeze(1)
        self.register_buffer("ul_flat_idx", ul_flat_idx, persistent=False)

    def forward(self, x):
        # x: (batch, N, N, num_features) — gather strict upper triangle, embed,
        # scatter back, then symmetrize (matches the original output exactly).
        B = x.shape[0]
        N = self.num_particles
        x_flat = x.reshape(B, N * N, self.in_dim)
        hidden = x_flat.index_select(1, self.ul_flat_idx)  # (B, L, in_dim)

        hidden = hidden.permute(0, 2, 1)  # (B, in_dim, L)
        for layer, norm in zip(self.conv_layers, self.norm_layers):
            hidden = layer(hidden)
            hidden = norm(hidden)
            hidden = self.activation(hidden)
        hidden = hidden.permute(0, 2, 1)  # (B, L, out_dim)

        out_flat = hidden.new_zeros(B, N * N, self.out_dim)
        out_flat = out_flat.index_copy(1, self.ul_flat_idx, hidden)
        output = out_flat.reshape(B, N, N, self.out_dim)
        return output + output.transpose(1, 2)


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - self.drop_prob

    def forward(self, x):
        if self.training:
            batch_size = x.shape[0]
            n_dims = len(x.shape)
            shape = (batch_size,) + (1,) * (n_dims - 1)
            random_tensor = self.keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            rand_binary_tensor = random_tensor.floor()
            return x / self.keep_prob * rand_binary_tensor
        return x
