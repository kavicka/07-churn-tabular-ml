"""Visual example: slide the alarm threshold over 1409 test customers and watch the cost.

Every dot is one test customer, placed by the model's churn probability (red = really churned).
Customers right of the line get a retention offer. The right panel shows what that costs.
Reads results.json (from churn.py), writes assets/threshold.gif.

Run:  uv run python threshold_gif.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

R = json.loads(Path("results.json").read_text())
F, C = R["final"], R["costs"]
p, y = np.array(F["test_proba"]), np.array(F["test_y"])
grid, test_cost = np.array(F["cost_curve"]["threshold"]), np.array(F["cost_curve"]["test"])
BLUE, RED, GRAY, DARK = "#2563eb", "#dc2626", "#94a3b8", "#1e293b"

# sweep up, back down, then settle on the chosen threshold and hold
path = np.concatenate([np.linspace(0.02, 0.95, 40), np.linspace(0.95, F["threshold"], 25), np.full(20, F["threshold"])])

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
jitter = np.random.default_rng(0).uniform(0, 1, len(p))
a1.scatter(p[y == 0], jitter[y == 0], s=6, color=GRAY, alpha=0.6, label="stayed")
a1.scatter(p[y == 1], jitter[y == 1], s=8, color=RED, alpha=0.8, label="churned")
shade = a1.axvspan(0.5, 1, color=BLUE, alpha=0.08)
line = a1.axvline(0.5, color=BLUE, lw=2)
a1.set(
    xlim=(0, 1),
    ylim=(-0.05, 1.05),
    yticks=[],
    xlabel="model's churn probability (calibrated LightGBM)",
    title=f"{len(p)} test customers · right of the line = gets a {C['offer']:.0f} € offer",
)
a1.legend(loc="upper center", fontsize=8, ncol=2)
info = a1.text(0.01, -0.33, "", transform=a1.transAxes, fontsize=9, family="monospace", va="top")

a2.plot(grid, test_cost, color=BLUE, lw=2)
for name, col in [("never offer", RED), ("offer everyone", "#ea580c")]:
    a2.axhline(R["policies"][name], color=col, ls=":", lw=1.2, label=f"{name}: {R['policies'][name]:.0f} €")
(dot,) = a2.plot([], [], "o", color=DARK, ms=9)
a2.set(xlim=(0, 1), xlabel="threshold", ylabel="cost per customer (€)", title="What it costs")
a2.legend(fontsize=8, loc="upper right")
a2.grid(alpha=0.3)
fig.subplots_adjust(bottom=0.3, left=0.04, right=0.98, wspace=0.18)


def frame(i: int):
    t = path[i]
    alarm = p >= t
    caught, wasted, missed = (
        int((alarm & (y == 1)).sum()),
        int((alarm & (y == 0)).sum()),
        int((~alarm & (y == 1)).sum()),
    )
    cost = float(
        np.where(alarm, C["offer"] + y * (1 - C["save_rate"]) * C["lost_customer"], y * C["lost_customer"]).mean()
    )
    line.set_xdata([t, t])
    shade.set_x(t)
    shade.set_width(1 - t)
    dot.set_data([t], [cost])
    best = " ← chosen from training data" if i >= len(path) - 20 else ""
    info.set_text(
        f"threshold {t:.2f}{best}\n"
        f"offers sent {alarm.sum():5d}   churners caught {caught:4d} / {int(y.sum())} (recall {caught / y.sum():.0%})\n"
        f"wasted offers {wasted:4d}   churners missed {missed:4d}   cost {cost:6.1f} € per customer"
    )
    return line, shade, dot, info


FuncAnimation(fig, frame, frames=len(path), blit=False).save("assets/threshold.gif", writer=PillowWriter(fps=8), dpi=80)
