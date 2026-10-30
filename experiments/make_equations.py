"""Render the display equations as high-resolution images.

The document previously carried these as native OMML. That is the better
format in principle -- it stays editable in Word -- but OMML is not rendered
by every consumer of a .docx, and in particular LibreOffice, which is what
produces the PDF here, drops it silently. Equations that vanish from the PDF
are worse than equations that cannot be edited, so they are rasterised.

Each equation is transcribed from the OMML in paper.js. The transcription is
the only place where the mathematics could be corrupted, so every expression
is written out in full below next to the number it carries in the paper, and
the rendered images are checked by eye after generation.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(exist_ok=True)

DPI = 600
FONTSIZE = 9.5

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
})

EQUATIONS: dict[int, str] = {
    # (1) observation vector
    1: r"$m(t) = [\, u(t),\; r(t),\; \lambda(t),\; \ell(t),\; N(t) \,]^{\mathsf{T}}$",

    # (2) utilisation law with safety margin
    2: r"$\tilde{N}(t+h) \;=\; \left\lceil\, (1+\gamma)\cdot"
       r"\frac{\hat{\lambda}(t+h)}{\mu\, u^{*}} \,\right\rceil$",

    # (3) binding requirement is the larger of the CPU and memory bounds
    3: r"$\tilde{N}(t+h) \;\leftarrow\; \max\left\{\, \tilde{N}_{\mathrm{CPU}}(t+h),\;"
       r"\left\lceil \hat{r}(t+h)\, N(t) / r^{*} \right\rceil \,\right\}$",

    # (4) Kingman heavy-traffic approximation for p95 latency
    4: r"$\hat{\ell} \;=\; \bar{s} \;+\; \frac{c_a^{2} + c_s^{2}}{2}\,"
       r"\frac{\rho}{1-\rho}\, \bar{s}, \qquad \rho \;=\; \hat{\lambda}/(N\mu)$",

    # (5) plant with cold-start dead time
    5: r"$N(t+1) = N(t) + u(t-\tau), \quad"
       r"\tau = \left\lceil \tau_c / \Delta t \right\rceil, \quad u \in \mathbb{Z}$",

    # (6) horizon-dead-time law
    6: r"$H\, \Delta t \;\geq\; \tau_c + \Delta t_{\mathrm{ctrl}}$",

    # (7) receding-horizon objective
    7: r"$J(u) = \sum_{h=1}^{H} \left[\, q_h \left( N(t+h) - \tilde{N}(t+h) \right)^{2}"
       r" + r\, u(t+h-1)^{2} + s \left( u(t+h-1) - u(t+h-2) \right)^{2} \,\right]$",

    # (8) asymmetric tracking weight
    8: r"$q_h = q^{+} \;\;\mathrm{if}\;\; N(t+h) < \tilde{N}(t+h), \;\;"
       r"\mathrm{else}\;\; q^{-}, \qquad q^{+} \gg q^{-}$",

    # (9) pool bounds, rate limit, integrality
    9: r"$N_{\min} \leq N(t+h) \leq N_{\max}, \quad |u| \leq \Delta_{\max},"
       r" \quad u \in \mathbb{Z}$",

    # (10) deadband and cooldown gates
    10: r"$u_{\mathrm{applied}}(t) = u^{*}(t) \;\;\mathrm{if}\;\; |u^{*}(t)| \geq \theta"
        r" \;\;\mathrm{and}\;\; t - t_{\mathrm{up}} \geq T_{\mathrm{cool}},"
        r" \;\;\mathrm{else}\;\; 0$",

    # (11) clipped-integral margin update
    11: r"$\gamma(t+1) = \mathrm{clip}\left( \gamma(t) + \kappa_{\gamma}\, \bar{e}(t),\;"
        r" \gamma_{\min},\; \gamma_{\max} \right)$",
}


def render(num: int, expr: str) -> tuple[str, float]:
    fig = plt.figure(figsize=(0.01, 0.01), dpi=DPI)
    fig.patch.set_alpha(0.0)
    t = fig.text(0, 0, expr, fontsize=FONTSIZE, color="#101418")
    name = f"eq{num:02d}.png"
    path = OUT / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.02,
                transparent=True)
    plt.close(fig)
    import struct
    with open(path, "rb") as fh:
        head = fh.read(32)
    w = struct.unpack(">I", head[16:20])[0]
    return name, w / DPI


if __name__ == "__main__":
    print(f"rendering {len(EQUATIONS)} equations at {DPI} dpi, {FONTSIZE} pt")
    for num in sorted(EQUATIONS):
        name, win = render(num, EQUATIONS[num])
        frame = "column" if win <= 3.0 else "spans both columns"
        print(f"  ({num:2d}) {name}  {win:5.2f} in   {frame}")
