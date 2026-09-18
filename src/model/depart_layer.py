import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.utils import (
    FFN,
    LayerScale,
    StochasticDepth,
    make_norm,
)


class TalkingMultiheadSelfAttention(nn.Module):
    """Multihead self-attention with talking heads, optionally biased by pairwise interaction features.

    use_interaction=False (decoder): no interaction projection is created and the `interaction`
    argument of forward is ignored."""

    def __init__(self, dim, num_heads, dropout=None, use_qknorm=False, use_interaction=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_interaction = use_interaction

        self.linear_qkv = nn.Linear(dim, dim * 3)
        self.linear_out = nn.Linear(dim, dim)

        self.linear_talking_1 = nn.Linear(num_heads, num_heads)
        self.linear_talking_2 = nn.Linear(num_heads, num_heads)
        self.linear_talking_int = nn.Linear(num_heads, num_heads) if use_interaction else None

        # Optional per-head QK-normalization (HyperScale-style): RMSNorm over the
        # head dimension applied to q and k before the QK^T product. Stabilizes
        # attention logits and pairs well with bf16 autocast.
        if use_qknorm:
            head_dim = dim // num_heads
            self.q_norm = nn.RMSNorm(head_dim)
            self.k_norm = nn.RMSNorm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        self.dropout = nn.Dropout(dropout) if dropout is not None else None
        self.attn_dropout = nn.Dropout(dropout) if dropout is not None else None

    def forward(self, x, interaction, mask):
        B, N, C = x.shape[0], x.shape[1], x.shape[2]

        qkv = self.linear_qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        scale = (q.shape[-1]) ** 0.5
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        attention_weights = attention_weights.permute(0, 2, 3, 1)
        attention_weights = self.linear_talking_1(attention_weights)
        attention_weights = attention_weights.permute(0, 3, 1, 2)

        if self.use_interaction:
            interaction = self.linear_talking_int(interaction)
            interaction = interaction.permute(0, 3, 1, 2)
            attention_weights = attention_weights + interaction

        attention_weights = attention_weights.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.linear_talking_2(attention_weights.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        # Re-zero padded key columns: linear_talking_2 has a bias, so it turns the
        # post-softmax ~0 weights on masked keys back into a nonzero b[h], leaking
        # padded particles into every token's output (the -inf mask alone is not
        # enough once a biased linear runs post-softmax). Same broadcast as above.
        attention_weights = attention_weights.masked_fill(mask.unsqueeze(1) == 0, 0.0)
        if self.attn_dropout is not None:
            attention_weights = self.attn_dropout(attention_weights)

        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).reshape(B, N, C)
        output = self.linear_out(output)
        if self.dropout is not None:
            output = self.dropout(output)

        return output, attention_weights


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        layer_scale_ini_val,
        expansion,
        drop_rate,
        layer_drop_rate,
        use_rmsnorm=False,
        use_qknorm=False,
        use_interaction=True,
    ):
        super().__init__()

        self.pre_mhsa_ln = make_norm(dim, use_rmsnorm)
        self.mhsa = TalkingMultiheadSelfAttention(
            dim, num_heads, drop_rate, use_qknorm=use_qknorm, use_interaction=use_interaction
        )
        self.post_mhsa_scale = LayerScale(layer_scale_ini_val, dim)
        self.post_mhsa_stoch_depth = StochasticDepth(layer_drop_rate)

        self.pre_ffn_ln = make_norm(dim, use_rmsnorm)
        self.ffn = FFN(dim, expansion, drop_rate, use_rmsnorm=use_rmsnorm)
        self.post_ffn_scale = LayerScale(layer_scale_ini_val, dim)
        self.post_ffn_stoch_depth = StochasticDepth(layer_drop_rate)

        # Normalizes the additive interaction-bias logits across the (tiny) head
        # axis, which can carry a meaningful mean -> keep LayerNorm (centering)
        # here even when the rest of the model uses RMSNorm.
        self.int_pre_msha_ln = nn.LayerNorm(num_heads) if use_interaction else None

    def forward(self, x, interaction, mask):
        attended = self.pre_mhsa_ln(x)
        if self.int_pre_msha_ln is not None:
            interaction = self.int_pre_msha_ln(interaction)
        attended, _ = self.mhsa(attended, interaction, mask)
        attended = self.post_mhsa_scale(attended)
        attended = self.post_mhsa_stoch_depth(attended)
        attended = x + attended

        ffned = self.pre_ffn_ln(attended)
        ffned = self.ffn(ffned)
        ffned = self.post_ffn_scale(ffned)
        ffned = self.post_ffn_stoch_depth(ffned)
        return attended + ffned


class SelfAttnDeParT(nn.Module):
    def __init__(
        self,
        dim,
        num_attn_layers,
        num_heads,
        expansion,
        layer_scale_ini_val,
        drop_rate,
        stochastic_depth_drop_rate,
        use_rmsnorm=False,
        use_qknorm=False,
        use_interaction=True,
    ):
        assert dim % 2 == 0, "dim must be even."
        super().__init__()

        self.self_attn_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim,
                    num_heads,
                    layer_scale_ini_val,
                    expansion,
                    drop_rate,
                    layer_drop_rate=self._stochastic_prob(
                        i, num_attn_layers, stochastic_depth_drop_rate
                    ),
                    use_rmsnorm=use_rmsnorm,
                    use_qknorm=use_qknorm,
                    use_interaction=use_interaction,
                )
                for i in range(num_attn_layers)
            ]
        )

    @staticmethod
    def _stochastic_prob(step, total_steps, drop_rate):
        return drop_rate * step / (total_steps - 1) if total_steps > 1 else 0.0

    def forward(self, x, interaction, mask):
        mask = mask.unsqueeze(1).expand(-1, x.shape[1], -1)
        hidden = x
        for block in self.self_attn_blocks:
            hidden = block(hidden, interaction, mask)
        return hidden


class MultiheadClassAttention(nn.Module):
    def __init__(self, dim, heads, dropout=None):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            dim, heads, dropout=(dropout if dropout is not None else 0.0), batch_first=True
        )

    def forward(self, x, class_token, mask):
        key_padding_mask = mask == 0
        output, _ = self.multihead_attn(
            query=class_token, key=x, value=x, key_padding_mask=key_padding_mask
        )
        return output


class ClassAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        layer_scale_init_value,
        expansion,
        dropout,
        layer_drop_rate,
        use_rmsnorm=False,
    ):
        super().__init__()

        self.pre_mhca_ln = make_norm(dim, use_rmsnorm)
        self.mhca = MultiheadClassAttention(dim, heads, dropout)
        self.post_mhca_scale = LayerScale(layer_scale_init_value, dim)
        self.post_mhca_stoch_depth = StochasticDepth(layer_drop_rate)

        self.pre_ffn_ln = make_norm(dim, use_rmsnorm)
        self.ffn = FFN(dim, expansion, dropout, use_rmsnorm=use_rmsnorm)
        self.post_ffn_scale = LayerScale(layer_scale_init_value, dim)
        self.post_ffn_stoch_depth = StochasticDepth(layer_drop_rate)

    def forward(self, x, class_token, mask):
        attended = torch.concat((class_token, x), dim=1)
        attended = self.pre_mhca_ln(attended)
        # Use the pre-normed class-token slice as the query so query and key/value
        # share the same (normed) scale — matching the self-attention house style
        # (raw class_token would put the QK^T logits on mismatched scales).
        attended = self.mhca(attended, attended[:, :1], mask)
        attended = self.post_mhca_scale(attended)
        attended = self.post_mhca_stoch_depth(attended)
        attended = attended + class_token

        ffned = self.pre_ffn_ln(attended)
        ffned = self.ffn(ffned)
        ffned = self.post_ffn_scale(ffned)
        ffned = self.post_ffn_stoch_depth(ffned)
        return attended + ffned


class ClassAttnDeParT(nn.Module):
    def __init__(
        self,
        dim,
        num_class_attn_layers,
        expansion,
        num_heads,
        layer_scale_ini_val,
        drop_rate=None,
        stochastic_depth_drop_rate=None,
        use_rmsnorm=False,
    ):
        super().__init__()

        drop_rate = drop_rate if drop_rate is not None else 0.0
        stochastic_depth_drop_rate = (
            stochastic_depth_drop_rate if stochastic_depth_drop_rate is not None else 0.0
        )

        self.class_attn_layers = nn.ModuleList(
            [
                ClassAttentionBlock(
                    dim,
                    num_heads,
                    layer_scale_ini_val,
                    expansion,
                    drop_rate,
                    layer_drop_rate=self._stochastic_prob(
                        i, num_class_attn_layers, stochastic_depth_drop_rate
                    ),
                    use_rmsnorm=use_rmsnorm,
                )
                for i in range(num_class_attn_layers)
            ]
        )

        self.class_token = nn.Parameter(torch.empty(1, 1, dim), requires_grad=True)
        nn.init.trunc_normal_(self.class_token, std=0.02, mean=0.0)

    @staticmethod
    def _stochastic_prob(step, total_steps, drop_rate):
        return drop_rate * step / (total_steps - 1) if total_steps > 1 else 0.0

    def forward(self, x, mask):
        hidden = x
        class_token = torch.tile(self.class_token, (x.shape[0], 1, 1))

        device = x.device
        mask = torch.concat(
            (torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=device), mask), dim=1
        )

        for layer in self.class_attn_layers:
            class_token = layer(hidden, class_token, mask)
        return class_token
