# Numerical check quoted in ../02_design_qa.md (2026-09-15).
# Run: /global/common/software/m3246/HAXAD/software/haxad3/bin/python presence_grad_check.py
import torch, torch.nn.functional as F
torch.manual_seed(0)
B, n_slots, n_feat = 256, 11, 7

# toy decoder outputs that require gradient: feature predictions and presence logits
x_hat   = torch.randn(B, n_slots, n_feat, requires_grad=True)
p_logit = (torch.randn(B, n_slots) * 0.5).requires_grad_(True)  # random logits around 0
x       = torch.randn(B, n_slots, n_feat)
present = (torch.rand(B, n_slots) < 0.4).float()                # true mask
valid   = torch.ones(n_slots, n_feat)

def report(name, loss):
    gx, gp = torch.autograd.grad(loss, [x_hat, p_logit], allow_unused=True)
    gp_norm = 0.0 if gp is None else gp.norm().item()
    gp_mean = float("nan") if gp is None else gp.mean().item()
    print(f"{name:42s} grad->features {gx.norm():8.3f}   grad->presence {gp_norm:8.3f}   mean sign {gp_mean:+.4f}")

# (1) hard cut: gate the MSE with (sigmoid > 0.5). Step function -> no gradient to presence.
gate_hard = (torch.sigmoid(p_logit) > 0.5).float()
report("hard cut  mask=(p>0.5)*MSE", ((x_hat - x)**2 * gate_hard[..., None]).mean())

# (2) soft gate: sigmoid(logit) * MSE. Gradient exists but is positive => descent lowers every logit
gate_soft = torch.sigmoid(p_logit)
report("soft gate sigmoid(p)*MSE", ((x_hat - x)**2 * gate_soft[..., None]).mean())

# (3) proposed: BCE on logits + MSE masked by the TRUE presence
w = present[..., None] * valid
mse = ((x_hat - x)**2 * w).sum() / w.sum()
bce = F.binary_cross_entropy_with_logits(p_logit, present)
report("proposed  BCE(p, m) + MSE*m_true", bce + mse)

# sign of the soft-gate gradient: d/dlogit [sigmoid(l) * mse] = sigmoid'(l) * mse >= 0 for every entry
gp_soft, = torch.autograd.grad(((x_hat - x)**2 * torch.sigmoid(p_logit)[..., None]).mean(), [p_logit])
print(f"\nsoft gate: fraction of presence-logit gradients that are positive = {(gp_soft > 0).float().mean():.3f}"
      "  (descent lowers every logit -> model learns to call objects absent)")
open_frac = (torch.sigmoid(p_logit) > 0.5).float().mean()
print(f"hard cut : gate open for {open_frac:.2f} of slots, yet presence head receives no gradient at all")
