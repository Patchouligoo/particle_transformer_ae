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
        output_size,
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
        self.output_classes = output_size
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

        self.self_attn_layer = SelfAttnDeParT(
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
        self.layernorm_selfattn = make_norm(self.embedding_dim, self.use_rmsnorm)

        self.class_attn_layer = ClassAttnDeParT(
            dim=self.embedding_dim,
            num_class_attn_layers=self.num_class_attn_layers,
            num_heads=self.num_heads,
            expansion=self.expansion,
            layer_scale_ini_val=self.layer_scale_ini_val,
            drop_rate=self.drop_rate,
            stochastic_depth_drop_rate=self.stochastic_depth_drop_rate,
            use_rmsnorm=self.use_rmsnorm,
        )
        self.layernorm_classattn = make_norm(self.embedding_dim, self.use_rmsnorm)

        self.output_layer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.output_classes),
        )

    def encode(self, x):
        node, interaction, mask = x

        node = self.node_embedding(node)
        interaction = self.interaction_embedding(interaction)

        node = self.self_attn_layer(node, interaction, mask)
        node = self.layernorm_selfattn(node)

        output = self.class_attn_layer(node, mask)
        output = self.layernorm_classattn(output)
        assert output.shape[1] == 1
        output = output[:, 0, :]

        output = self.output_layer(output)
        
        return output

    def forward(self, x):
        z = self.encode(x)

        return z


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
