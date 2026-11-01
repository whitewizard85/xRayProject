"""
make_figures.py -- figure del paper dai soli file aggregati di analysis/ (nessun dato VinDr grezzo).

    python make_figures.py <cartella analysis> <cartella output>

Fig. 1  variabilita' tra seed: AUC interna (NIH) vs esterna (VinDr) per cella del fattoriale
Fig. 2  calibrazione per loss: ECE raw / temperature / Platt, NIH e VinDr (+ intercetta)
Fig. 3  effetti di crop e loss sulla localizzazione (VinDr), schedule collassato: 40 unita'
Fig. 4  accordo tra mappe (stesso setup/seed diverso vs un fattore diverso) e sanity check
"""
import sys
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # palette validata (dataviz)
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e4"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans", "savefig.dpi": 300})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


R = pd.read_csv(os.path.join(A, "runs_table.csv"))
F = R[R.block == "factorial"].copy()
F["cell"] = ("crop" + F.crop.astype(str) + "·" + F.loss.str.upper() + "·" + F.schedule + "·" +
             np.where(F.aug == 1, "flip", "noaug"))

# ---------------------------------------------------------------- Fig. 1 (schedule collassato: 8 celle x 5 seed)
F1 = F.groupby(["crop", "loss", "aug", "seed"])[["nih_test_macro_auc", "vindr_test_macro_auc"]].mean().reset_index()
F1["cell"] = (np.where(F1.crop == 1, "center crop", "resize") + " · " + F1.loss.str.upper() + " · " +
              np.where(F1.aug == 1, "flip", "no aug."))
cells = F1.groupby("cell").vindr_test_macro_auc.std().sort_values().index.tolist()[::-1]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
for ax, col, title in [(axes[0], "nih_test_macro_auc", "Internal test (NIH)"),
                       (axes[1], "vindr_test_macro_auc", "External test (VinDr)")]:
    mid = (F1[col].min() + F1[col].max()) / 2
    for i, c in enumerate(cells):
        g = F1[F1.cell == c]
        color = C1 if g.aug.iloc[0] == 1 else C2
        ax.plot([g[col].min(), g[col].max()], [i, i], color=color, lw=2, alpha=0.35, solid_capstyle="round")
        ax.scatter(g[col], np.full(len(g), i), s=16, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlim(mid - 0.035, mid + 0.035)        # stessa ampiezza (0.07) nei due pannelli
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel("Macro ROC-AUC (each dot = one seed)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_yticks(range(len(cells)))
axes[0].set_yticklabels(cells, fontsize=6.8)
axes[0].set_ylim(-0.6, len(cells) - 0.4)
sd_n = F1.groupby("cell").nih_test_macro_auc.std().mean()
sd_v = F1.groupby("cell").vindr_test_macro_auc.std().mean()
fig.suptitle(f"Seed-to-seed spread (mean within-configuration SD): {sd_n:.4f} internal vs {sd_v:.4f} external",
             x=0.01, ha="left", fontsize=8.5, color=INK)
fig.text(0.01, -0.03, "Each configuration averaged over the two learning-rate schedules (near-identical models, see text). "
         "Blue = horizontal-flip augmentation, orange = none. Both panels span the same AUC range.",
         fontsize=6.5, color=MUTED)
fig.tight_layout()
save(fig, "fig1_seed_variability")

# ---------------------------------------------------------------- Fig. 2
C = pd.read_csv(os.path.join(A, "calibration_by_loss.csv"))
C = C[C.arch == "convnext_base"]
losses = ["bce", "focal", "asl", "wbce"]
names = {"bce": "BCE", "focal": "Focal", "asl": "ASL", "wbce": "wBCE"}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
for ax, ds, title in [(axes[0], "nih_test", "Internal test (NIH)"), (axes[1], "vindr_test", "External test (VinDr)")]:
    x = np.arange(len(losses))
    for j, (rc, col, lab) in enumerate([("raw", C2, "uncalibrated"), ("ts", C4, "temperature scaling"),
                                        ("platt", C1, "Platt scaling")]):
        s = C[(C.dataset == ds) & (C.recal == rc)].set_index("loss").reindex(losses)
        ax.bar(x + (j - 1) * 0.26, s.ece_mean, 0.24, yerr=s.ece_sd, color=col, label=lab,
               error_kw=dict(lw=0.8, capsize=1.5, ecolor=MUTED), edgecolor="white", linewidth=0.6)
    ax.set_xticks(x, [names[l] for l in losses])
    ax.set_title(title, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Macro ECE (10 occupancy-weighted bins)")
axes[0].legend(frameon=False, loc="upper left")
fig.text(0.01, -0.04, "Recalibration maps fitted on NIH validation only; bars = mean over 5 seeds, whiskers = SD. "
         "Reference cell (no crop, cosine 10 epochs, flip).", fontsize=6.5, color=MUTED)
fig.tight_layout()
save(fig, "fig2_calibration")

# ---------------------------------------------------------------- Fig. 3 (schedule collassato)
mets = [("loc_vindr_gradcam_pointing", "Grad-CAM", "pointing game"),
        ("loc_vindr_gradcam_energy", "Grad-CAM", "energy-in-box"),
        ("loc_vindr_gradcam_norm_energy", "Grad-CAM", "area-normalized energy"),
        ("loc_vindr_gradcampp_pointing", "Grad-CAM++", "pointing game"),
        ("loc_vindr_gradcampp_energy", "Grad-CAM++", "energy-in-box"),
        ("loc_vindr_gradcampp_norm_energy", "Grad-CAM++", "area-normalized energy"),
        ("loc_vindr_ig_pointing", "IG", "pointing game"),
        ("loc_vindr_ig_energy", "IG", "energy-in-box"),
        ("loc_vindr_ig_norm_energy", "IG", "area-normalized energy")]
G = F.groupby(["crop", "loss", "aug", "seed"])[[m for m, _, _ in mets]].mean().reset_index()
G["crop_c"] = G.crop.map({0: -.5, 1: .5}); G["loss_c"] = G.loss.map({"bce": -.5, "asl": .5})
G["aug_c"] = G.aug.map({0: -.5, 1: .5})
rows = []
for m, meth, met in mets:
    sd_runs = G[m].std()
    fit = smf.ols(f"{m} ~ crop_c*loss_c*aug_c", data=G).fit()
    ci = fit.conf_int()
    for t, lab in [("crop_c", "center crop"), ("loss_c", "ASL vs BCE")]:
        # standardizzato sulla SD dei residui (= variabilita' tra seed) per confrontare metriche diverse
        s = np.sqrt(fit.scale)
        rows.append(dict(method=meth, metric=met, factor=lab, d=fit.params[t] / s,
                         lo=ci.loc[t, 0] / s, hi=ci.loc[t, 1] / s, raw=fit.params[t], p=fit.pvalues[t]))
E = pd.DataFrame(rows)
E.to_csv(os.path.join(OUT, "fig3_effects_table.csv"), index=False)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
labels = [f"{r.method} · {r.metric}" for _, r in E[E.factor == "center crop"].iterrows()]
y = np.arange(len(labels))[::-1]
for off, (fac, col) in zip([0.17, -0.17], [("center crop", C1), ("ASL vs BCE", C2)]):
    s = E[E.factor == fac]
    ax.hlines(y + off, s.lo, s.hi, color=col, lw=1.6)
    ax.scatter(s.d, y + off, s=16, color=col, edgecolor="white", linewidth=0.6, zorder=3, label=fac)
ax.axvline(0, color=MUTED, lw=0.8)
for k in (3, 6):
    ax.axhline(len(labels) - k - 0.5, color=GRID, lw=0.8)
ax.set_yticks(y, labels)
ax.set_xlabel("Effect on VinDr localization, in units of seed-to-seed SD (95% CI)")
ax.legend(frameon=False, loc="lower right")
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
save(fig, "fig3_localization_effects")

# ---------------------------------------------------------------- Fig. 4
Ag = pd.read_csv(os.path.join(A, "attribution_agreement.csv"))
Ag = Ag[(Ag.dataset == "vindr") & ~Ag.comparison.str.contains("secondary") & ~Ag.comparison.str.contains("schedule")]
order = ["within (seed only)", "between: crop (1 vs 0)", "between: aug (flip vs none)", "between: loss (ASL vs BCE)"]
short = {"within (seed only)": "same setup,\nother seed", "between: crop (1 vs 0)": "+ crop\nchanged",
         "between: aug (flip vs none)": "+ augment.\nchanged", "between: loss (ASL vs BCE)": "+ loss\nchanged"}
san = {m: R[f"sanity_vindr_{m}_rho"].mean() for m in ["gradcam", "gradcampp", "ig"]}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(width_ratios=[2.2, 1]))
ax = axes[0]
x = np.arange(len(order))
for j, (m, col, lab) in enumerate([("gradcam", C1, "Grad-CAM"), ("gradcampp", C3, "Grad-CAM++"), ("ig", C2, "IG")]):
    s = Ag[Ag.method == m].set_index("comparison").reindex(order)
    ax.bar(x + (j - 1) * 0.26, s.mean_spearman, 0.24, color=col, label=lab, edgecolor="white", linewidth=0.6,
           yerr=[s.mean_spearman - s.ci_lo, s.ci_hi - s.mean_spearman], error_kw=dict(lw=0.8, ecolor=MUTED))
ax.set_xticks(x, [short[o] for o in order])
ax.set_ylabel("Spearman ρ between maps\n(same image–finding pair)")
ax.set_ylim(0, 1)
ax.set_title("Agreement between models (VinDr)", color=INK, loc="left")
ax.legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.0))
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax = axes[1]
ms = ["gradcam", "gradcampp", "ig"]
ax.bar(range(3), [san[m] for m in ms], 0.6, color=[C1, C3, C2], edgecolor="white", linewidth=0.6)
for i, m in enumerate(ms):
    ax.text(i, san[m] + 0.02, f"{san[m]:.2f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(range(3), ["Grad-CAM", "Grad-CAM++", "IG"])
ax.set_ylim(0, 1)
ax.set_title("Randomization test", color=INK, loc="left")
ax.set_ylabel("ρ original vs randomized\n(lower = model-sensitive)")
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig4_agreement_sanity")
print("figure salvate in", OUT)
