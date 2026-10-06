"""Two-slope ("hinge") regression with bootstrap uncertainty, in PyTorch.

    y(x) = b + k1 (x - xb) + (k2 - k1) * softplus(beta (x - xb)) / beta

i.e. slope k1 before the break point xb and k2 after it, joined smoothly over
~1/beta. Used to locate where an observable stops following the charge, e.g.
where the Ni K-edge stops shifting while Na is still being extracted.

All bootstrap replicates are fitted as one batch: parameters have shape
(B, 4) and a single autograd graph optimises them together, so 2000
replicates cost about as much as a handful of sequential fits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def _hinge(x, b, k1, k2, xb, beta):
    u = x - xb
    return b + k1 * u + (k2 - k1) * torch.nn.functional.softplus(beta * u) / beta


def _grid_init(x: torch.Tensor, y: torch.Tensor, grid: torch.Tensor):
    """Exact least squares of the sharp hinge for every candidate xb.

    x: (N,), y: (B, N), grid: (G,). Returns best (b, k1, k2, xb) per replicate.
    """
    u = x[None, :] - grid[:, None]                              # (G, N)
    X = torch.stack([torch.ones_like(u), torch.clamp(u, max=0), torch.clamp(u, min=0)], -1)  # (G,N,3)
    pinv = torch.linalg.pinv(X)                                  # (G,3,N)
    coef = torch.einsum("gpn,bn->bgp", pinv, y)                  # (B,G,3)
    pred = torch.einsum("gnp,bgp->bgn", X, coef)
    sse = ((pred - y[:, None, :]) ** 2).sum(-1)                  # (B,G)
    j = sse.argmin(1)
    c = coef[torch.arange(len(j)), j]
    return c[:, 0], c[:, 1], c[:, 2], grid[j]


def fit_hinge_batch(x: np.ndarray, Y: np.ndarray, beta: float = 200.0, steps: int = 400,
                    margin: int = 2, lr: float = 0.02):
    """Fit one hinge per row of ``Y`` (B, N) against shared ``x`` (N,).

    The break point is kept inside the data with at least ``margin`` points on
    each side. Returns a dict of numpy arrays (b, k1, k2, xb, sse).
    """
    dt = torch.float64
    xs = torch.as_tensor(x, dtype=dt)
    order = torch.argsort(xs)
    xs = xs[order]
    ys = torch.as_tensor(np.atleast_2d(Y), dtype=dt)[:, order]
    # standardise for well-conditioned optimisation
    xm, xsd = xs.mean(), xs.std()
    ym, ysd = ys.mean(1, keepdim=True), ys.std(1, keepdim=True).clamp_min(1e-12)
    xn, yn = (xs - xm) / xsd, (ys - ym) / ysd
    lo, hi = xn[margin - 1], xn[-margin]
    grid = torch.linspace(float(lo), float(hi), 200, dtype=dt)
    b, k1, k2, xb = _grid_init(xn, yn, grid)
    params = torch.stack([b, k1, k2, xb], 1).clone().requires_grad_(True)
    opt = torch.optim.Adam([params], lr=lr)
    bn = beta * float(xsd)
    for _ in range(steps):
        opt.zero_grad()
        p = params
        pred = _hinge(xn[None, :], p[:, :1], p[:, 1:2], p[:, 2:3], p[:, 3:4], bn)
        loss = ((pred - yn) ** 2).sum(1)
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            params[:, 3].clamp_(float(lo), float(hi))
    with torch.no_grad():
        p = params
        pred = _hinge(xn[None, :], p[:, :1], p[:, 1:2], p[:, 2:3], p[:, 3:4], bn)
        sse = (((pred - yn) * ysd) ** 2).sum(1)
        ysd1 = ysd[:, 0]
        out = {
            "xb": (p[:, 3] * xsd + xm).numpy(),
            "k1": (p[:, 1] * ysd1 / xsd).numpy(),
            "k2": (p[:, 2] * ysd1 / xsd).numpy(),
            "b": (p[:, 0] * ysd1 + ym[:, 0]).numpy(),
            "sse": sse.numpy(),
        }
    return out


@dataclass
class HingeResult:
    xb: float
    xb_ci: tuple
    k1: float
    k2: float
    b: float
    sse: float
    sse_line: float
    n: int
    boot_xb: np.ndarray

    @property
    def delta_bic(self) -> float:
        """BIC(line) - BIC(hinge); > 6 is strong evidence for a break."""
        n = self.n
        bic = lambda sse, k: n * np.log(sse / n) + k * np.log(n)
        return float(bic(self.sse_line, 2) - bic(self.sse, 4))

    def predict(self, x, beta: float = 200.0):
        x = torch.as_tensor(np.asarray(x, float))
        return _hinge(x, self.b, self.k1, self.k2, self.xb, beta).numpy()


def fit_hinge(x, y, n_boot: int = 2000, seed: int = 0, ci: float = 0.95, **kw) -> HingeResult:
    """Best fit plus a wild (Rademacher) residual bootstrap of the break point."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    best = fit_hinge_batch(x, y[None], **kw)
    res = HingeResult(xb=float(best["xb"][0]), xb_ci=(np.nan, np.nan), k1=float(best["k1"][0]),
                      k2=float(best["k2"][0]), b=float(best["b"][0]), sse=float(best["sse"][0]),
                      sse_line=float(((np.polyval(np.polyfit(x, y, 1), x) - y) ** 2).sum()),
                      n=len(x), boot_xb=np.array([]))
    yhat = res.predict(x)
    r = y - yhat
    r = r * np.sqrt(len(x) / max(len(x) - 4, 1))           # small-sample inflation
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_boot, len(x)))
    boot = fit_hinge_batch(x, yhat[None] + signs * r[None], **kw)
    q = (1 - ci) / 2
    res.boot_xb = boot["xb"]
    res.xb_ci = (float(np.quantile(boot["xb"], q)), float(np.quantile(boot["xb"], 1 - q)))
    return res
