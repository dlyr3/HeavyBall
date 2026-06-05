"""
Diagnostic ADOPT : on fait tourner le MEME probleme jouet avec TROIS optimiseurs
  (1) ADOPT de REFERENCE, traduit directement du papier (notre verite terrain)
  (2) heavyball FUSED       (update_by_adopt)  -> le chemin BUGUE de la classe ADOPT
  (3) heavyball NON-FUSED   (scale_by_adopt)   -> le chemin SAIN, qui applique le momentum
et on trace les trajectoires pour voir QUI colle a la reference et QUI derive.
"""

import torch
import matplotlib.pyplot as plt

import heavyball.utils
import heavyball.chainable as C
heavyball.utils.compile_mode = None  # pas de compilateur C sur cette machine -> mode eager
from heavyball import ADOPT as HeavyballADOPT

# ---------------------------------------------------------------------------
# Hyperparametres communs aux trois cotes : meme lr, memes betas, meme eps.
# ---------------------------------------------------------------------------
LR = 0.01
BETA1 = 0.9
BETA2 = 0.99
EPS = 1e-8
N_STEPS = 400

# Probleme jouet le plus simple possible : minimiser  f(x) = 0.5 * ||x||^2
# Le gradient vaut donc exactement g = x. Convexe, optimum en 0.
INIT = torch.tensor([2.0, -1.0, 0.5], dtype=torch.float32)


def loss_of(x):
    return 0.5 * (x ** 2).sum()


# ---------------------------------------------------------------------------
# (1) ADOPT de REFERENCE, ecrit a la main d'apres le papier.
# ---------------------------------------------------------------------------
class ReferenceADOPT:
    def __init__(self, param, lr, beta1, beta2, eps):
        self.p = param
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = torch.zeros_like(param)   # premier moment (momentum)
        self.v = torch.zeros_like(param)   # second moment
        self.t = 0

    @torch.no_grad()
    def step(self):
        g = self.p.grad
        if self.t == 0:
            # premier pas : pas de "v precedent" -> on initialise v = g^2
            self.v = g * g
        else:
            denom = self.v.sqrt().clamp(min=self.eps)   # racine du v PRECEDENT
            normed = g / denom                          # gradient normalise AVANT le momentum
            self.m = self.beta1 * self.m + (1 - self.beta1) * normed
            self.p.add_(self.m, alpha=-self.lr)         # on deplace le parametre
            self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)   # v mis a jour APRES
        self.t += 1


# ---------------------------------------------------------------------------
# (3) heavyball NON-FUSED : meme classe que ADOPT, mais auto_fuse=False pour
#     empecher la conversion automatique scale_by_adopt -> update_by_adopt.
#     On emprunte donc reellement le chemin scale_by_adopt (le sain).
# ---------------------------------------------------------------------------
class NonFusedADOPT(C.BaseOpt):
    auto_fuse = False

    def __init__(self, params, lr=0.0025, betas=(0.9, 0.99), eps=1e-8, weight_decay=0,
                 warmup_steps=0, multi_tensor=True, storage_dtype="float32", mars=False,
                 caution=False, mars_gamma=0.0025, gradient_clipping=None,
                 update_clipping=None, palm=False, beta2_scale=0.8,
                 compile_step=False, promote=False, ecc=None, param_ecc=None,
                 orig_shapes=None, **kwargs):
        params, defaults = C._build_defaults(locals())
        super().__init__(params, defaults, gradient_clipping, update_clipping, palm,
                         fns=(C.scale_by_adopt,))


HEAVYBALL_KW = dict(
    lr=LR, betas=(BETA1, BETA2), eps=EPS, weight_decay=0, warmup_steps=0,
    caution=False, mars=False, palm=False, gradient_clipping=None,
    update_clipping=None, compile_step=False,
)


def run(make_opt, is_reference=False):
    """Fait tourner un optimiseur N_STEPS fois et renvoie (trajectoire des poids, loss)."""
    x = INIT.clone().requires_grad_(True)
    opt = make_opt(x)
    weights, losses = [x.detach().clone()], []
    for _ in range(N_STEPS):
        if is_reference:
            if x.grad is not None:
                x.grad.zero_()
        else:
            opt.zero_grad()
        l = loss_of(x)
        l.backward()
        losses.append(l.item())
        opt.step()
        weights.append(x.detach().clone())
    return torch.stack(weights).numpy(), losses


if __name__ == "__main__":
    runs = {
        "Reference (paper)": run(lambda x: ReferenceADOPT(x, LR, BETA1, BETA2, EPS), is_reference=True),
        "heavyball FUSED (buggy)": run(lambda x: HeavyballADOPT([x], **HEAVYBALL_KW)),
        "heavyball NON-FUSED (correct)": run(lambda x: NonFusedADOPT([x], **HEAVYBALL_KW)),
    }
    styles = {
        "Reference (paper)": dict(color="black", lw=2.5, ls="-"),
        "heavyball FUSED (buggy)": dict(color="crimson", lw=2, ls="--"),
        "heavyball NON-FUSED (correct)": dict(color="royalblue", lw=2, ls=":"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("ADOPT: reference vs heavyball fused vs heavyball non-fused", fontsize=14, weight="bold")

    # three panels: one weight component per panel
    for comp, ax in zip(range(3), axes.flat):
        for label, (weights, _) in runs.items():
            ax.plot(weights[:, comp], label=label, **styles[label])
        ax.axhline(0, color="grey", lw=0.8, alpha=0.6)  # the optimum is at 0
        ax.set_title(f"weight component {comp}  (start = {INIT[comp].item()})")
        ax.set_xlabel("step")
        ax.set_ylabel("weight value")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # fourth panel: the loss (log scale to see convergence)
    ax = axes.flat[3]
    for label, (_, losses) in runs.items():
        ax.plot(losses, label=label, **styles[label])
    ax.set_yscale("log")
    ax.set_title("loss over steps (log scale)")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "adopt_diagnostic.png"
    fig.savefig(out, dpi=130)
    print(f"figure enregistree -> {out}")