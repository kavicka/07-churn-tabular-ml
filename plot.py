"""Turn results.json into the plots used in the README."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = json.loads(Path("results.json").read_text())
Path("assets").mkdir(exist_ok=True)
BLUE, RED, GREEN, GRAY, ORANGE, DARK = "#2563eb", "#dc2626", "#16a34a", "#94a3b8", "#ea580c", "#1e293b"
MODEL_COLORS = [GRAY, "#a16207", "#7c3aed", GREEN, "#60a5fa", BLUE]


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(f"assets/{name}", dpi=150)
    plt.close(fig)


def eda() -> None:
    e = R["eda"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for a, (key, title) in zip(
        axes,
        [
            ("churn_by_contract", "Contract"),
            ("churn_by_tenure_months", "Tenure (months)"),
            ("churn_by_internet", "Internet service"),
        ],
    ):
        d = e[key]
        names = list(d)
        rates = [v["rate"] if isinstance(v, dict) else v for v in d.values()]
        bars = a.bar(names, rates, color=BLUE, width=0.6)
        a.axhline(e["churn_rate"], color=GRAY, ls="--", lw=1, label=f"all customers {e['churn_rate']:.1%}")
        for b, r in zip(bars, rates):
            a.text(b.get_x() + b.get_width() / 2, r + 0.01, f"{r:.0%}", ha="center", fontsize=9)
        a.set(title=f"Churn rate by {title.lower()}", xlabel=title)
        a.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("share who churned")
    axes[0].legend(fontsize=9)
    fig.suptitle(f"EDA: {e['customers']} customers, {e['churn_rate']:.1%} churned", fontsize=12)
    save(fig, "eda.png")


def models() -> None:
    m = R["models"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={"width_ratios": [1.1, 1.2, 0.9]})
    a = axes[0]
    for (name, c), col in zip(R["pr_curves"].items(), MODEL_COLORS):
        if name.startswith("naive"):
            a.axhline(m[name]["test_pr_auc"], color=col, ls="--", lw=1, label=f"{name}: {m[name]['test_pr_auc']:.3f}")
        else:
            a.plot(
                c["recall"],
                c["precision"],
                color=col,
                lw=2 if "Optuna" in name else 1.3,
                label=f"{name}: {m[name]['test_pr_auc']:.3f}",
            )
    a.set(
        xlabel="recall (churners caught)",
        ylabel="precision (offers that hit a churner)",
        ylim=(0, 1.02),
        title="Precision–recall on the test set (legend: PR-AUC)",
    )
    a.legend(fontsize=7.5, loc="upper right")
    a.grid(alpha=0.3)

    a = axes[1]
    names = list(m)
    y = np.arange(len(names))
    a.barh(
        y + 0.2,
        [m[n]["cv_pr_auc_mean"] for n in names],
        0.4,
        xerr=[m[n]["cv_pr_auc_std"] for n in names],
        color=GRAY,
        label="5-fold CV PR-AUC (± std)",
        capsize=3,
    )
    a.barh(y - 0.2, [m[n]["test_pr_auc"] for n in names], 0.4, color=BLUE, label="test PR-AUC")
    a.plot([m[n]["test_accuracy_at_0.5"] for n in names], y, "D", color=ORANGE, ms=6, label="test accuracy at 0.5")
    a.set_yticks(y, names)
    a.invert_yaxis()
    a.set(xlim=(0, 1), xlabel="score", title="Accuracy barely separates the models; PR-AUC does")
    a.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)
    a.grid(alpha=0.3, axis="x")

    a = axes[2]
    t = R["optuna"]["trial_cv_pr_auc"]
    a.plot(range(1, len(t) + 1), t, "o", color=GRAY, ms=4, label="trial")
    a.plot(range(1, len(t) + 1), np.maximum.accumulate(t), color=BLUE, lw=2, label="best so far")
    a.axhline(m["LightGBM (default)"]["cv_pr_auc_mean"], color=ORANGE, ls="--", lw=1, label="LightGBM default")
    a.set(xlabel="Optuna trial", ylabel="CV PR-AUC", title=f"Optuna: {len(t)} trials")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    save(fig, "models.png")


def leakage() -> None:
    lk = R["leakage"]
    fig, a = plt.subplots(figsize=(8.5, 4.4))
    groups = ["PR-AUC", "ROC-AUC"]
    vals = {
        "CV, oversampled BEFORE split (leak)": [lk["wrong"]["cv_pr_auc"], lk["wrong"]["cv_roc_auc"]],
        "CV, oversampled inside each fold": [lk["right"]["cv_pr_auc"], lk["right"]["cv_roc_auc"]],
        "real test set": [lk["wrong"]["test_pr_auc"], lk["wrong"]["test_roc_auc"]],
    }
    x = np.arange(len(groups))
    for i, ((lab, v), col) in enumerate(zip(vals.items(), [RED, GREEN, BLUE])):
        bars = a.bar(x + (i - 1) * 0.27, v, 0.25, color=col, label=lab)
        for b, val in zip(bars, v):
            a.text(b.get_x() + b.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", fontsize=8)
    a.set_xticks(x, groups)
    a.set(
        ylim=(0, 1.2),
        ylabel="score (LightGBM default)",
        title=f"Leakage: {lk['wrong']['val_churners_with_copy_in_train']:.0%} of validation churners had an exact copy in training",
    )
    a.legend(fontsize=8, loc="upper center", ncol=3)
    a.grid(alpha=0.3, axis="y")
    save(fig, "leakage.png")


def cost() -> None:
    f, c, pol = R["final"], R["final"]["cost_curve"], R["policies"]
    fig, a = plt.subplots(figsize=(9, 4.6))
    a.plot(c["threshold"], c["train_oof"], color=GRAY, lw=1.5, label="training data (out-of-fold), used to choose")
    a.plot(c["threshold"], c["test"], color=BLUE, lw=2, label="test set")
    for name, col in [("never offer", RED), ("offer everyone", ORANGE), ("perfect knowledge", GREEN)]:
        a.axhline(pol[name], color=col, ls=":", lw=1.2, label=f"{name}: {pol[name]:.0f} €")
    a.axvline(f["threshold"], color=DARK, lw=1, label=f"chosen threshold {f['threshold']:.2f}")
    a.axvline(
        R["costs"]["theory_threshold"],
        color=DARK,
        ls="--",
        lw=1,
        label=f"theory OFFER / (save · LOST) = {R['costs']['theory_threshold']:.2f}",
    )
    a.set(
        xlabel="send an offer when P(churn) ≥ threshold",
        ylabel="cost per customer (€)",
        title="Expected cost vs threshold (offer 100 €, lost customer 500 €, offer saves 50 %)",
    )
    a.legend(fontsize=7.5, loc="upper right")
    a.grid(alpha=0.3)
    save(fig, "cost.png")


def calibration() -> None:
    fig, a = plt.subplots(figsize=(6.4, 5.6))
    a.plot([0, 1], [0, 1], "k:", lw=1, label="perfectly calibrated")
    for (name, d), col in zip(R["calibration"].items(), [GREEN, BLUE, "#60a5fa", RED]):
        a.plot(
            d["curve"]["predicted"],
            d["curve"]["observed"],
            "o-",
            color=col,
            ms=4,
            lw=1.5,
            label=f"{name} (Brier {d['brier']:.3f})",
        )
    a.set(
        xlabel="predicted churn probability (10 bins)",
        ylabel="share that actually churned",
        title="Calibration on the test set",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    a.legend(fontsize=7.5, loc="upper left")
    a.grid(alpha=0.3)
    save(fig, "calibration.png")


def shap_plot() -> None:
    s = R["shap"]
    names = s["features"][::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1, 1.3]})
    y = np.arange(len(names))
    share = np.array(s["mean_abs_shap"][::-1]) / sum(s["mean_abs_shap"])
    a1.barh(y + 0.2, share, 0.4, color=BLUE, label="SHAP: share of mean |SHAP| (top 12)")
    a1.barh(y - 0.2, s["split_share"][::-1], 0.4, color=GRAY, label="LightGBM: share of splits")
    a1.set_yticks(y, names)
    a1.set(xlabel="share of importance", title="Same model, two importance rankings")
    a1.legend(fontsize=8, loc="lower right")
    a1.grid(alpha=0.3, axis="x")

    sv = np.array(s["beeswarm"]["shap"])[:, ::-1]
    val = np.array(s["beeswarm"]["value"])[:, ::-1]
    rng = np.random.default_rng(0)
    for j in range(len(names)):
        v = val[:, j]
        colour = (v - v.min()) / (np.ptp(v) or 1)  # low value = blue, high = red
        sc = a2.scatter(sv[:, j], j + rng.uniform(-0.3, 0.3, len(v)), c=colour, cmap="coolwarm", s=5, alpha=0.6)
    a2.axvline(0, color="k", lw=0.8)
    a2.set_yticks(y, names)
    a2.set(
        xlabel="SHAP value (log-odds of churn): > 0 pushes towards churn",
        title="Every test customer: how each feature moved their prediction",
    )
    cb = fig.colorbar(sc, ax=a2, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("feature value")
    save(fig, "shap.png")


if __name__ == "__main__":
    eda()
    models()
    leakage()
    cost()
    calibration()
    shap_plot()
