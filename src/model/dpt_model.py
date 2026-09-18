import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.model.utils import (
    CNNEmbedding,
    FCEmbedding,
    build_interaction_torch,
    make_norm,
)
from src.model.depart_layer import (
    SelfAttnDeParT,
    ClassAttnDeParT,
)


class DPTModel(nn.Module):
    def __init__(
        self,
        input_shape,
        latent_dim,
        num_self_attn_layers=4,
        num_class_attn_layers=2,
        num_heads=8,
        expansion=3,
        layer_scale_ini_val=1,
        drop_rate=0.2,
        stochastic_depth_drop_rate=0.2,
        embedding_dim=128,
        num_embeding_layers=2,
        conv_embed_latent_dim=128,
        num_conv_layers=2,
        use_rmsnorm=False,
        use_qknorm=False,
        interaction_norm_type="batchnorm",
    ):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_self_attn_layers = num_self_attn_layers
        self.num_class_attn_layers = num_class_attn_layers
        self.num_heads = num_heads
        self.expansion = expansion
        self.layer_scale_ini_val = layer_scale_ini_val
        self.drop_rate = drop_rate
        self.stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self.use_rmsnorm = use_rmsnorm
        self.use_qknorm = use_qknorm
        self.interaction_norm_type = interaction_norm_type

        self.num_particles = input_shape[0][0]
        self.num_particle_features = input_shape[0][1]
        self.num_interaction_features = input_shape[1][2]

        self.embedding_dim = embedding_dim
        self.num_embeding_layers = num_embeding_layers
        self.node_embedding = FCEmbedding(
            self.num_particle_features, self.embedding_dim, self.num_embeding_layers
        )
        # Slot (position) embedding: one learned vector per slot, added to every event's token of that
        # slot. Attention alone is permutation-equivariant, so without it the encoder cannot tell an
        # electron from a photon or a muon with the same kinematics; the slots are typed and pt-ordered,
        # so row i means the same object in every event. Replaces HAXAD's per-object type one-hot.
        self.slot_embedding = nn.Parameter(torch.empty(self.num_particles, self.embedding_dim))
        nn.init.trunc_normal_(self.slot_embedding, std=0.02)


        # ----------------------- encoder layers -----------------------
        self.conv_embed_latent_dim = conv_embed_latent_dim
        self.num_conv_layers = num_conv_layers
        self.interaction_embedding = CNNEmbedding(
            self.num_conv_layers,
            self.conv_embed_latent_dim,
            self.num_interaction_features,
            self.num_heads,
            self.num_particles,
            norm_type=interaction_norm_type,
        )

        self.encoder_self_attn_layer = SelfAttnDeParT(
            dim=self.embedding_dim,
            num_attn_layers=self.num_self_attn_layers,
            num_heads=self.num_heads,
            expansion=self.expansion,
            layer_scale_ini_val=self.layer_scale_ini_val,
            drop_rate=self.drop_rate,
            stochastic_depth_drop_rate=self.stochastic_depth_drop_rate,
            use_rmsnorm=self.use_rmsnorm,
            use_qknorm=self.use_qknorm,
        )
        self.encoder_layernorm_selfattn = make_norm(self.embedding_dim, self.use_rmsnorm)

        self.encoder_class_attn_layer = ClassAttnDeParT(
            dim=self.embedding_dim,
            num_class_attn_layers=self.num_class_attn_layers,
            num_heads=self.num_heads,
            expansion=self.expansion,
            layer_scale_ini_val=self.layer_scale_ini_val,
            drop_rate=self.drop_rate,
            stochastic_depth_drop_rate=self.stochastic_depth_drop_rate,
            use_rmsnorm=self.use_rmsnorm,
        )
        self.encoder_layernorm_classattn = make_norm(self.embedding_dim, self.use_rmsnorm)

        self.encode_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.latent_dim),
        )

        # ----------------------- decoder layers -----------------------
        self.decoder_input_decompression = nn.Sequential(
            nn.Linear(self.latent_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

        self.decoder_self_attn_layer = SelfAttnDeParT(
            dim=self.embedding_dim,
            num_attn_layers=self.num_self_attn_layers,
            num_heads=self.num_heads,
            expansion=self.expansion,
            layer_scale_ini_val=self.layer_scale_ini_val,
            drop_rate=self.drop_rate,
            stochastic_depth_drop_rate=self.stochastic_depth_drop_rate,
            use_rmsnorm=self.use_rmsnorm,
            use_qknorm=self.use_qknorm,
            use_interaction=False,  # the decoder has no pairwise features; slot tokens attend to each other only
        )
        self.decoder_layernorm_selfattn = make_norm(self.embedding_dim, self.use_rmsnorm)

        self.output_projection_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.num_particle_features),
        )

        # ------------------ presence layer ------------------
        self.presence_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.num_particles),
        )


    def encode(self, x):

        # --------------- encode ---------------
        node, interaction, mask = x

        node = self.node_embedding(node) + self.slot_embedding  # (B, 13, d) + (13, d)
        interaction = self.interaction_embedding(interaction)

        node = self.encoder_self_attn_layer(node, interaction, mask)
        node = self.encoder_layernorm_selfattn(node)

        z = self.encoder_class_attn_layer(node, mask)
        z = self.encoder_layernorm_classattn(z)
        assert z.shape[1] == 1
        z = z[:, 0, :]

        z = self.encode_layer(z)

        return z

    def decode(self, z):

        # during decoding, no mask is needed, so we create one with all ones
        mask = torch.ones(z.size(0), self.num_particles, device=z.device, dtype=torch.float32)

        # --------------- decode ---------------
        z_decompressed = self.slot_embedding + self.decoder_input_decompression(z).unsqueeze(1)  # (B, 13, d)
        node = self.decoder_self_attn_layer(z_decompressed, interaction=None, mask=mask)
        node = self.decoder_layernorm_selfattn(node)

        reco_output = self.output_projection_layer(node)

        return reco_output

    def forward(self, x):
        z = self.encode(x)
        reco_output = self.decode(z)
        presence = self.presence_layer(z)

        return reco_output, presence, z


class DPTDataSet(Dataset):
    """Pre-loads all tensors onto the target device for fast batch access.

    Interaction features are NOT stored (they dominate memory at ~726 floats/
    event). Instead the raw kinematics ``kin_raw`` (N, NUM_PARTICLES, 4) are kept
    and the dense (B, N, N, 6) interaction tensor is rebuilt per batch on-device
    via ``build_interaction_torch`` — numerically identical to the precomputed
    path but ~5x lighter in resident memory.
    """

    def __init__(
        self, particle_level, kin_raw, mask_data, label, device, interaction_norm
    ):
        self.device = device
        self.interaction_norm = interaction_norm

        self.particle_level_tensor = torch.tensor(particle_level, dtype=torch.float32).to(device)
        self.mask_tensor = torch.tensor(mask_data, dtype=torch.float32).to(device)
        self.kin_raw_tensor = torch.tensor(kin_raw, dtype=torch.float32).to(device)

        self.label = torch.tensor(label, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.particle_level_tensor)

    def __getitem__(self, idx):
        # Single-sample path (legacy DataLoader): add/remove a batch dim around
        # the builder, which expects (B, N, 4).
        interaction = build_interaction_torch(
            self.kin_raw_tensor[idx].unsqueeze(0), self.interaction_norm
        ).squeeze(0)
        return (
            self.particle_level_tensor[idx],
            interaction,
            self.mask_tensor[idx],
            self.label[idx],
        )

    def get_batch(self, idx):
        """Gather a whole batch with one indexing op per field.

        `idx` is a 1-D sequence/tensor of sample indices. The dataset already
        lives on `self.device`, so this avoids the per-sample __getitem__ +
        collate path (thousands of tiny GPU kernels per batch). The interaction
        tensor is built for the batch from the gathered raw kinematics.
        """
        idx = torch.as_tensor(idx, device=self.device, dtype=torch.long)
        interaction = build_interaction_torch(self.kin_raw_tensor[idx], self.interaction_norm)
        return (
            self.particle_level_tensor[idx],
            interaction,
            self.mask_tensor[idx],
            self.label[idx],
        )


class GpuBatchLoader:
    """Yields whole batches by indexing the device-resident tensors of a DPTDataSet once per batch
    (torch DataLoader would build the interaction tensor one event at a time: 30-500x slower here).
    Each batch is the 4-tuple of DPTDataSet.get_batch: particle_level, interaction, mask, label."""

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset, self.batch_size, self.shuffle, self.drop_last = dataset, batch_size, shuffle, drop_last

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        order = torch.randperm(n, device=self.dataset.device) if self.shuffle else torch.arange(n, device=self.dataset.device)
        if self.drop_last:
            order = order[: len(self) * self.batch_size]
        for idx in order.split(self.batch_size):
            yield self.dataset.get_batch(idx)
