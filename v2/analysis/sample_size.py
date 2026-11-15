"""
sample_size.py -- how many independently trained models per arm are needed to detect a difference in
matched-class AUC, given the seed-to-seed variability observed in the factorial block.

    python sample_size.py <runs_table.csv> <out_dir>

Inputs: runs_table.csv from 06_analyze.py (factorial block; the two learning-rate schedules of each seed and
configuration are averaged, as in the paper). Seed SD is pooled over the eight configurations (32 df).
Two designs: (a) independent seeds per arm, two-sample t test; (b) the same seeds in both arms, paired by seed,
using the residual SD after removing the seed component shared across configurations.
Two-sided alpha = .05, power = .80, exact t quantiles (iterative). Output: sample_size.csv, fig5_sample_size.pdf.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats


def n_two_sample(sig, delta, alpha=.05, power=.8):
    n = 2
    while True:
        df = 2 * n - 2
        q = stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)
        if n >= 2 * q ** 2 * sig ** 2 / delta ** 2:
            return n
        n += 1


def n_paired(sd_diff, delta, alpha=.05, power=.8):
    n = 2
    while True:
        df = n - 1
        q = stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)
        if n >= q ** 2 * sd_diff ** 2 / delta ** 2:
            return n
        n += 1


def main(runs_csv, out):
    R = pd.read_csv(runs_csv)
    F = R[R["block"] == "factorial"].copy()
    F["cfg"] = F["crop"].astype(str) + "_" + F["loss"] + "_" + F["aug"].astype(str)
    U = F.groupby(["cfg", "seed"])[["nih_test_matched_auc", "vindr_test_matched_auc"]].mean().reset_index()
    rows, curves = [], {}
    deltas = np.round(np.arange(0.004, 0.0301, 0.001), 4)
    for col, lab in [("nih_test_matched_auc", "internal"), ("vindr_test_matched_auc", "external")]:
        W = U.pivot(index="seed", columns="cfg", values=col)
        resid = W.sub(W.mean(axis=0), axis=1)
        seed_eff = resid.mean(axis=1)
        share = float((seed_eff ** 2).sum() * W.shape[1] / (resid ** 2).values.sum())
        e = resid.sub(seed_eff, axis=0)
        sd_res = float(np.sqrt((e ** 2).values.sum() / ((W.shape[0] - 1) * (W.shape[1] - 1))))
        df = W.shape[1] * (W.shape[0] - 1)
        sd = float(np.sqrt(W.var(axis=0).mean()))
        lo = sd * np.sqrt(df / stats.chi2.ppf(.975, df))
        hi = sd * np.sqrt(df / stats.chi2.ppf(.025, df))
        for d in [0.005, 0.01, 0.02]:
            rows.append(dict(test_set=lab, delta_auc=d, seed_sd=sd, seed_sd_ci_lo=lo, seed_sd_ci_hi=hi, df=df,
                             shared_seed_share=share, residual_sd_after_seed_block=sd_res,
                             n_per_arm_independent=n_two_sample(sd, d),
                             n_per_arm_independent_ci=f"{n_two_sample(lo, d)}-{n_two_sample(hi, d)}",
                             n_per_arm_shared_seeds=n_paired(np.sqrt(2) * sd_res, d)))
        curves[lab] = ([n_two_sample(sd, d) for d in deltas], [n_paired(np.sqrt(2) * sd_res, d) for d in deltas],
                       [n_two_sample(lo, d) for d in deltas], [n_two_sample(hi, d) for d in deltas])
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "sample_size.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ind, pai, lo, hi = curves["external"]
    ax.fill_between(deltas, lo, hi, color="#d95f02", alpha=.15, lw=0)
    ax.plot(deltas, ind, color="#d95f02", lw=2, label="External test, independent seeds per arm")
    ax.plot(deltas, pai, color="#d95f02", lw=2, ls="--", label="External test, same seeds in both arms (paired)")
    ax.plot(deltas, curves["internal"][0], color="#1b9e77", lw=2, label="Internal test, independent seeds per arm")
    ax.axhline(5, color="0.4", lw=1, ls=":")
    ax.text(deltas[-1], 3.6, "5 models per arm (this study)", ha="right", va="top", fontsize=8, color="0.3")
    ax.set_yscale("log")
    ax.set_xlabel("Difference in matched-class AUC to detect")
    ax.set_ylabel("Models per arm (80% power, α = .05)")
    ax.set_xlim(deltas[0], deltas[-1])
    ax.grid(True, which="both", color="0.92")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_sample_size.pdf"))
    fig.savefig(os.path.join(out, "fig5_sample_size.png"), dpi=200)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
