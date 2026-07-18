"""
Generate the project figures from the reported cross-validation results.

These values are the ACTUAL outputs of lapse_model.py on the 1,773-customer
cohort (33.3% lapse). If you re-run with different data or RECALL_TARGET,
update the dictionaries below or regenerate them from the script's printout.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

INK = "#1b2a4a"
BLUE = "#2f6fb0"
ORANGE = "#e08a1e"
RED = "#c0433a"
GREEN = "#3a8a5f"
GREY = "#9aa4b2"

# ---------------------------------------------------------------- 1. model comparison
metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
logreg = [0.662, 0.495, 0.793, 0.610, 0.760]
logreg_e = [0.021, 0.020, 0.017, 0.020, 0.029]
rf = [0.704, 0.549, 0.625, 0.584, 0.753]
rf_e = [0.015, 0.021, 0.026, 0.019, 0.020]
baseline = [0.667, 0.000, 0.000, 0.000, 0.500]

x = np.arange(len(metrics))
w = 0.38
fig, ax = plt.subplots(figsize=(8.4, 4.6))
ax.bar(x - w/2, logreg, w, yerr=logreg_e, capsize=3, label="Logistic Regression",
       color=BLUE, edgecolor="white")
ax.bar(x + w/2, rf, w, yerr=rf_e, capsize=3, label="Random Forest",
       color=ORANGE, edgecolor="white")
for i, b in enumerate(baseline):
    ax.hlines(b, x[i]-w, x[i]+w, color=GREY, linestyle="--", lw=1.6,
              label="Baseline (majority class)" if i == 0 else None)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score (5-fold CV mean +/- SD)")
ax.set_title("Model performance vs. baseline\nLogistic and RF tie on AUC; the split is only the operating threshold",
             fontsize=11.5, color=INK, loc="left")
ax.legend(frameon=False, ncol=1, loc="upper right", fontsize=9.5)
fig.tight_layout(); fig.savefig("model_comparison.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 2. coefficients
names = ["frequency", "item_diversity", "monetary_total", "recency_days",
         "tip_rate", "tenure_days", "avg_order_value", "modifiers_per_order",
         "delivery_share", "avg_items_per_order"]
vals = [-2.312, -0.398, 0.264, 0.218, -0.106, -0.063, 0.055, 0.033, 0.026, -0.017]
order = np.argsort(np.abs(vals))
names_s = [names[i] for i in order]
vals_s = [vals[i] for i in order]
colors = [RED if v > 0 else GREEN for v in vals_s]

fig, ax = plt.subplots(figsize=(8.4, 4.8))
ax.barh(names_s, vals_s, color=colors, edgecolor="white")
ax.axvline(0, color=INK, lw=1)
ax.set_xlabel("Standardized logistic coefficient")
ax.set_title("What predicts lapse\nred = pushes toward lapsing   |   green = pushes toward staying",
             fontsize=11.5, color=INK, loc="left")
for y, v in enumerate(vals_s):
    ax.text(v + (0.04 if v >= 0 else -0.04), y, f"{v:+.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
ax.margins(x=0.18)
fig.tight_layout(); fig.savefig("coefficients.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 3. threshold trade-off
thr = [0.30, 0.40, 0.50, 0.60, 0.70]
prec = [0.433, 0.460, 0.495, 0.557, 0.592]
rec = [0.929, 0.876, 0.793, 0.649, 0.327]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.plot(thr, rec, "-o", color=BLUE, label="Recall (lapsers caught)")
ax.plot(thr, prec, "-o", color=ORANGE, label="Precision (flags that are right)")
ax.axvline(0.573, color=GREY, ls="--", lw=1.4)
ax.text(0.573, 0.05, " chosen = 0.57\n (recall >= 0.70)", color=INK, fontsize=8.5, va="bottom")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score (out-of-fold)")
ax.set_ylim(0, 1.0)
ax.set_title("Precision-recall trade-off\nlower threshold catches more lapsers but flags more false alarms",
             fontsize=11.5, color=INK, loc="left")
ax.legend(frameon=False, loc="center right", fontsize=9.5)
fig.tight_layout(); fig.savefig("threshold_tradeoff.png", bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- 4. revenue concentration
fig, ax = plt.subplots(figsize=(7.2, 4.2))
groups = ["Share of\ncustomers", "Share of\nrevenue"]
atrisk = [42.9, 16.7]
rest = [57.1, 83.3]
ax.bar(groups, atrisk, color=RED, label="Flagged at-risk", edgecolor="white")
ax.bar(groups, rest, bottom=atrisk, color=GREY, label="Not flagged", edgecolor="white")
for i, v in enumerate(atrisk):
    ax.text(i, v/2, f"{v:.1f}%", ha="center", va="center", color="white", fontweight="bold")
for i, (a, r) in enumerate(zip(atrisk, rest)):
    ax.text(i, a + r/2, f"{r:.1f}%", ha="center", va="center", color=INK)
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
ax.set_ylim(0, 100)
ax.set_title("Lapse risk concentrates in low-value customers\n43% of customers at risk, but only 17% of revenue",
             fontsize=11.5, color=INK, loc="left")
ax.legend(frameon=False, loc="lower right", fontsize=9.5)
fig.tight_layout(); fig.savefig("revenue_concentration.png", bbox_inches="tight"); plt.close(fig)

print("wrote 4 figures")
