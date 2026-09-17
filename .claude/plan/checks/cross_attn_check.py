# Numerical check quoted in ../02_design_qa.md (2026-09-15).
# Run: /global/common/software/m3246/HAXAD/software/haxad3/bin/python cross_attn_check.py
import torch, torch.nn as nn
torch.manual_seed(0)
B, n_slots, d, k = 4, 11, 64, 6

z = torch.randn(B, k)
to_d = nn.Linear(k, d)
memory = to_d(z).unsqueeze(1)                    # z as a single key/value token: (B, 1, d)
Q = torch.randn(1, n_slots, d)
h = Q + memory                                   # slot tokens (B, 11, d)
h_other = torch.randn(B, n_slots, d)             # completely different queries

xattn = nn.MultiheadAttention(d, num_heads=8, batch_first=True)
with torch.no_grad():
    out, w = xattn(query=h, key=memory, value=memory)
    out_other, _ = xattn(query=h_other, key=memory, value=memory)
    # closed form: out = W_o (W_v memory + b_v) + b_o, independent of the query
    W_v = xattn.in_proj_weight[2*d:]; b_v = xattn.in_proj_bias[2*d:]
    closed = xattn.out_proj(memory @ W_v.T + b_v).expand(-1, n_slots, -1)

print("attention weights, all entries         :", w.unique().tolist())
print("max |out_i - out_0| across slots       : %.2e" % (out - out[:, :1]).abs().max())
print("max |out(h) - out(random queries)|     : %.2e" % (out - out_other).abs().max())
print("max |out - W_o W_v z' closed form|     : %.2e" % (out - closed).abs().max())
