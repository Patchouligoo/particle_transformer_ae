# Numerical check quoted in ../02_design_qa.md (2026-09-15).
# Run: /global/common/software/m3246/HAXAD/software/haxad3/bin/python slot_query_check.py
import torch, torch.nn as nn
torch.manual_seed(0)

class SlotDecoder(nn.Module):
    def __init__(self, n_slots=11, n_feat=7, latent_dim=6, d=64, n_layers=3, n_heads=8, use_query=True):
        super().__init__()
        q = torch.randn(1, n_slots, d) * 0.02 if use_query else torch.zeros(1, n_slots, d)
        self.slot_query = nn.Parameter(q, requires_grad=use_query)
        self.from_latent = nn.Linear(latent_dim, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 3 * d, batch_first=True, norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.feat_head = nn.Linear(d, n_feat)
    def forward(self, z):
        h = self.slot_query + self.from_latent(z).unsqueeze(1)
        return self.feat_head(self.blocks(h))

z = torch.randn(4, 6)
for use_query in (False, True):
    torch.manual_seed(1)
    dec = SlotDecoder(use_query=use_query).eval()
    with torch.no_grad():
        out = dec(z)                                   # (4, 11, 7)
    spread = (out - out[:, :1, :]).abs().max().item()  # max difference between slot 0 and any other slot
    print(f"use_query={use_query!s:5}  output shape={tuple(out.shape)}  max |x_hat_i - x_hat_0| over slots = {spread:.3e}")
