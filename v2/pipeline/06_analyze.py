"""
06_analyze.py -- analisi statistica di tutti i run completati (anche parziali).

    python 06_analyze.py

Output in <work_dir>/analysis/:
  runs_table.csv                 una riga per run: fattori + tutte le metriche a livello di run
  summary_by_condition.csv       media, SD, n (seed) per condizione
  factorial_effects.csv          effetti principali e interazioni del fattoriale 2^4 (OLS su metriche
                                 a livello di RUN: il run e' l'unita' statistica -> niente pseudoreplicazione)
  factorial_variance.csv         quota di varianza spiegata da ciascun termine vs residuo (= variabilita' tra seed)
  localization_hboot.csv         effetti principali sulla localizzazione con bootstrap GERARCHICO
                                 (ricampiona seed dentro ogni cella E immagini, appaiato tra run)
  attribution_agreement.csv      accordo (Spearman) tra mappe: stesso setup/seed diverso vs un solo fattore diverso
  calibration_by_loss.csv        raw / temperature / Platt, NIH test e VinDr, per loss
  faithfulness_sanity.csv        deletion gain e test di randomizzazione per metodo
  fig_*.png, REPORT.md           figure e riepilogo leggibile

Rilanciabile in qualsiasi momento: usa solo i run gia' completati.
"""
import os
import json
import argparse
import itertools
import warnings

import numpy as np
import pandas as pd

from common import load_config, run_dir
from experiments import build_plan, FACTOR_LEVELS

warnings.filterwarnings("ignore")
FACTORS = list(FACTOR_LEVELS)
CODE = {"crop": {0: -0.5, 1: 0.5}, "loss": {"bce": -0.5, "asl": 0.5},
        "schedule": {"fx10": -0.5, "es20": 0.5}, "aug": {0: -0.5, 1: 0.5}}
# Classi comuni NIH/VinDr (one-to-one) per il confronto interno/esterno
MATCHED = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Pneumothorax",
           "Consolidation", "Fibrosis", "Pleural_Thickening"]
# IG, faithfulness e sanity sono riportati SOLO su VinDr (su NIH i sottoinsiemi fissi hanno pochissime coppie)
NIH_EXCLUDED_METHODS = {"ig"}

LEVEL_TXT = {"crop": "crop (1 vs 0)", "loss": "loss (ASL vs BCE)", "schedule": "schedule (es20 vs fx10)",
             "aug": "aug (flip vs none)"}


def load_runs(cfg):
    rows, attr = [], []
    for r in build_plan(cfg):
        rd = os.path.join(cfg["paths"]["work_dir"], "runs", r["run_id"])
        ev = os.path.join(rd, "eval.json")
        if not os.path.exists(ev):
            continue
        row = dict(r)
        row.update({k: v for k, v in json.load(open(ev)).items() if k not in r})
        pc_p = os.path.join(rd, "eval_perclass.csv")
        if os.path.exists(pc_p):
            pc = pd.read_csv(pc_p)
            for ds in ["nih_test", "vindr_test"]:
                sub = pc[(pc["dataset"] == ds) & (pc["target"].isin(MATCHED))]
                if len(sub):
                    row[f"{ds}_matched_auc"] = float(sub["auc"].mean())
        tp = os.path.join(rd, "train_done.json")
        if os.path.exists(tp):
            t = json.load(open(tp))
            row.update(best_val_auc=t["best_val_macro_auc"], epochs_run=t["epochs_run"], train_hours=t["total_hours"])
        ap = os.path.join(rd, "attr_pairs.csv")
        if os.path.exists(os.path.join(rd, "attr_done.json")) and os.path.exists(ap):
            a = pd.read_csv(ap)
            a = a[~((a["dataset"] == "nih") & (a["method"].isin(NIH_EXCLUDED_METHODS)))]
            if len(a):
                attr.append(a)
                g = a.groupby(["dataset", "method"])[["pointing", "energy", "norm_energy"]].mean()
                for (ds, m), v in g.iterrows():
                    for k in v.index:
                        row[f"loc_{ds}_{m}_{k}"] = v[k]
            fp = os.path.join(rd, "attr_faithfulness.csv")
            if os.path.exists(fp) and os.path.getsize(fp) > 5:
                f = pd.read_csv(fp)
                f = f[f["dataset"] == "vindr"]
                for (ds, m), v in f.groupby(["dataset", "method"])["faith_gain"].mean().items():
                    row[f"faith_{ds}_{m}_gain"] = v
            sp = os.path.join(rd, "attr_sanity.csv")
            if os.path.exists(sp) and os.path.getsize(sp) > 5:
                s = pd.read_csv(sp)
                s = s[s["pair_id"].astype(str).str.startswith("vindr|")]
                for m, v in s.groupby("method")["spearman_orig_vs_random"].mean().items():
                    row[f"sanity_vindr_{m}_rho"] = v
        rows.append(row)
    return pd.DataFrame(rows), (pd.concat(attr, ignore_index=True) if attr else pd.DataFrame())


def metric_cols(R):
    keys = ["nih_test_macro_auc", "vindr_test_macro_auc", "nih_test_matched_auc", "vindr_test_matched_auc",
            "nih_test_macro_prauc", "vindr_test_macro_prauc"]
    keys += [c for c in R.columns if c.startswith(("nih_test_raw_", "nih_test_ts_", "nih_test_platt_",
                                                   "vindr_test_raw_", "vindr_test_ts_", "vindr_test_platt_",
                                                   "loc_", "faith_", "sanity_"))]
    return [k for k in keys if k in R.columns]


# ------------------------- fattoriale (livello run) -------------------------
def factorial_ols(F, metrics):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    F = F.copy()
    for f in FACTORS:
        F[f + "_c"] = F[f].map(CODE[f])
    cells = F.groupby(FACTORS).size()
    full = len(cells) == 16 and cells.min() >= 2
    if full:
        rhs = "crop_c*loss_c*schedule_c*aug_c"
    elif len(F) >= 16 and len(cells) == 16:
        rhs = "(crop_c+loss_c+schedule_c+aug_c)**2"
    else:
        rhs = "crop_c+loss_c+schedule_c+aug_c"
    eff, var = [], []
    for m in metrics:
        d = F[[m] + [f + "_c" for f in FACTORS]].dropna()
        if len(d) < 6 or d[m].std() == 0:
            continue
        try:
            fit = smf.ols(f"Q('{m}') ~ {rhs}", data=d).fit()
        except Exception:
            continue
        ci = fit.conf_int()
        for term in fit.params.index:
            if term == "Intercept":
                continue
            eff.append(dict(metric=m, term=term.replace("_c", ""), estimate=fit.params[term],
                            ci_lo=ci.loc[term, 0], ci_hi=ci.loc[term, 1], p=fit.pvalues[term],
                            n_runs=int(fit.nobs), resid_sd_seed=float(np.sqrt(fit.scale)), model=rhs))
        try:
            an = sm.stats.anova_lm(fit, typ=2)
            tot = an["sum_sq"].sum()
            for term, ss in an["sum_sq"].items():
                var.append(dict(metric=m, term=term.replace("_c", ""), share=ss / tot))
        except Exception:
            pass
    return pd.DataFrame(eff), pd.DataFrame(var)


# --------------- bootstrap gerarchico sulla localizzazione ------------------
def localization_hboot(F, A, n_boot, ci, rng):
    out = []
    if A.empty:
        return pd.DataFrame()
    A = A[A["run_id"].isin(F["run_id"])]
    runs = F.set_index("run_id")
    for (ds, meth), g in A.groupby(["dataset", "method"]):
        for metric in ["pointing", "energy", "norm_energy"]:
            M = g.pivot_table(index="run_id", columns="pair_id", values=metric, aggfunc="mean")
            if M.shape[0] < 4:
                continue
            pair_ids = M.columns.to_numpy()
            img = np.array([p.split("|")[1] for p in pair_ids])
            uimg, inv = np.unique(img, return_inverse=True)
            X = M.to_numpy(dtype=float)
            V = ~np.isnan(X)
            X0 = np.where(V, X, 0.0)
            rr = runs.loc[M.index]
            cell_key = rr[FACTORS].astype(str).agg("|".join, axis=1).to_numpy()
            cells = {c: np.where(cell_key == c)[0] for c in np.unique(cell_key)}
            lev = {f: rr[f].map(CODE[f]).to_numpy() for f in FACTORS}

            def effects(wimg, run_sel):
                w = wimg[inv]
                rm = (X0 * w).sum(1) / np.maximum((V * w).sum(1), 1e-12)
                cm = {c: rm[idx].mean() for c, idx in run_sel.items()}
                res = {}
                for f in FACTORS:
                    hi = [cm[c] for c, idx in cells.items() if lev[f][idx[0]] > 0]
                    lo = [cm[c] for c, idx in cells.items() if lev[f][idx[0]] < 0]
                    res[f] = (np.mean(hi) - np.mean(lo)) if hi and lo else np.nan
                return res

            point = effects(np.ones(len(uimg)), cells)
            boots = {f: [] for f in FACTORS}
            for _ in range(n_boot):
                wimg = rng.multinomial(len(uimg), np.full(len(uimg), 1 / len(uimg))).astype(float)
                sel = {c: rng.choice(idx, size=len(idx), replace=True) for c, idx in cells.items()}
                e = effects(wimg, sel)
                for f in FACTORS:
                    boots[f].append(e[f])
            q = (1 - ci) / 2
            for f in FACTORS:
                b = np.array(boots[f], dtype=float)
                b = b[~np.isnan(b)]
                if not len(b):
                    continue
                out.append(dict(dataset=ds, method=meth, metric=metric, factor=LEVEL_TXT[f], effect=point[f],
                                ci_lo=np.quantile(b, q), ci_hi=np.quantile(b, 1 - q),
                                ci_excludes_zero=bool(np.quantile(b, q) > 0 or np.quantile(b, 1 - q) < 0),
                                n_runs=M.shape[0], n_pairs=M.shape[1], n_images=len(uimg)))
    return pd.DataFrame(out)


# ------------------------- accordo tra mappe --------------------------------
def standardized_ranks(maps):
    """maps (P, g, g) con NaN per coppie non calcolate -> (P, g*g) ranghi standardizzati."""
    P = maps.shape[0]
    X = maps.reshape(P, -1).astype(np.float32)
    ok = ~np.isnan(X).any(1)
    Z = np.full_like(X, np.nan)
    if ok.any():
        r = X[ok].argsort(1).argsort(1).astype(np.float32)
        r -= r.mean(1, keepdims=True)
        r /= np.maximum(r.std(1, keepdims=True), 1e-6)
        Z[ok] = r
    return Z


def attribution_agreement(cfg, F, n_boot, ci, rng, g_small=32):
    if F.empty:
        return pd.DataFrame()
    runs = F.set_index("run_id")
    cache = {}

    def get(run_id, meth):
        k = (run_id, meth)
        if k not in cache:
            p = os.path.join(cfg["paths"]["work_dir"], "runs", run_id, "attr_maps.npz")
            if not os.path.exists(p):
                cache[k] = None
            else:
                d = np.load(p)
                m = d.get(f"map_{meth}") if f"map_{meth}" in d.files else None
                if m is None:
                    cache[k] = None
                else:
                    m = m.astype(np.float32)
                    G = m.shape[-1]
                    f = G // g_small
                    m = m.reshape(m.shape[0], g_small, f, g_small, f).mean((2, 4)) if f > 1 else m
                    cache[k] = (d["pair_id"], standardized_ranks(m))
        return cache[k]

    ids = [r for r in F["run_id"] if os.path.exists(os.path.join(cfg["paths"]["work_dir"], "runs", r, "attr_maps.npz"))]
    rr = runs.loc[ids]
    out = []
    methods = cfg["attribution"]["methods"]
    # Confronto PRIMARIO: entrambe le famiglie usano seed DIVERSI, cosi' "between - within"
    # isola l'effetto del fattore. Le coppie a STESSO seed condividono inizializzazione e ordine
    # dei dati (somiglianza gonfiata): riportate solo come analisi secondaria.
    comps = {"within (seed only)": []}
    for f in FACTORS:
        comps[f"between: {LEVEL_TXT[f]}"] = []
        comps[f"between, same seed (secondary): {LEVEL_TXT[f]}"] = []
    for a_, b_ in itertools.combinations(ids, 2):
        A, B = rr.loc[a_], rr.loc[b_]
        diff = [f for f in FACTORS if A[f] != B[f]]
        if not diff and A["seed"] != B["seed"]:
            comps["within (seed only)"].append((a_, b_))
        elif len(diff) == 1 and A["seed"] != B["seed"]:
            comps[f"between: {LEVEL_TXT[diff[0]]}"].append((a_, b_))
        elif len(diff) == 1:
            comps[f"between, same seed (secondary): {LEVEL_TXT[diff[0]]}"].append((a_, b_))
    for meth in methods:
        per = {}
        pair_ids = None
        for name, plist in comps.items():
            vals = []
            for a_, b_ in plist:
                ga, gb = get(a_, meth), get(b_, meth)
                if ga is None or gb is None:
                    continue
                pair_ids = ga[0]
                vals.append(np.mean(ga[1] * gb[1], axis=1))      # Spearman per coppia immagine-target
            if vals:
                per[name] = np.vstack(vals)                     # (n_runpairs, P)
        if not per:
            continue
        ds = np.array([p.split("|")[0] for p in pair_ids])
        for d in np.unique(ds):
            if d == "nih" and meth in NIH_EXCLUDED_METHODS:
                continue
            sel = ds == d
            for name, Mx in per.items():
                Y = Mx[:, sel]
                colmean = np.nanmean(Y, 0)
                ok = ~np.isnan(colmean)
                if not ok.any():
                    continue
                cm = colmean[ok]
                bs = [np.mean(rng.choice(cm, size=len(cm), replace=True)) for _ in range(n_boot)]
                q = (1 - ci) / 2
                out.append(dict(dataset=d, method=meth, comparison=name, mean_spearman=float(np.mean(cm)),
                                ci_lo=float(np.quantile(bs, q)), ci_hi=float(np.quantile(bs, 1 - q)),
                                n_run_pairs=Y.shape[0], n_pairs=int(ok.sum())))
        cache = {k: v for k, v in cache.items() if k[1] != meth}
    return pd.DataFrame(out)


# ------------------------------- figure ------------------------------------
def figures(R, cal, agr, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not cal.empty:
        for ds in ["nih_test", "vindr_test"]:
            sub = cal[cal["dataset"] == ds]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 3.5))
            losses = sub["loss"].unique().tolist()
            x = np.arange(len(losses))
            for j, m in enumerate(["raw", "ts", "platt"]):
                s = sub[sub["recal"] == m].set_index("loss").reindex(losses)
                ax.bar(x + (j - 1) * 0.27, s["ece_mean"], 0.27, yerr=s["ece_sd"], label=m, capsize=2)
            ax.set_xticks(x, losses)
            ax.set_ylabel("macro ECE (10 bin)")
            ax.set_title(f"Calibrazione per loss - {ds}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"fig_calibration_{ds}.png"), dpi=200)
            plt.close(fig)
    if not agr.empty:
        for d, g in agr.groupby("dataset"):
            fig, ax = plt.subplots(figsize=(7, 3.8))
            g = g[~g["comparison"].str.contains("secondary")]
            comps = g["comparison"].unique().tolist()
            meths = g["method"].unique().tolist()
            x = np.arange(len(comps))
            for j, m in enumerate(meths):
                s = g[g["method"] == m].set_index("comparison").reindex(comps)
                ax.bar(x + (j - (len(meths) - 1) / 2) * 0.25, s["mean_spearman"], 0.25,
                       yerr=[s["mean_spearman"] - s["ci_lo"], s["ci_hi"] - s["mean_spearman"]], label=m, capsize=2)
            ax.set_xticks(x, comps, rotation=20, ha="right", fontsize=8)
            ax.set_ylabel("Spearman tra mappe")
            ax.set_title(f"Accordo delle attribuzioni ({d})")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"fig_agreement_{d}.png"), dpi=200)
            plt.close(fig)
    if "vindr_test_macro_auc" in R and R["vindr_test_macro_auc"].notna().any():
        fig, ax = plt.subplots(figsize=(4.5, 4))
        for b, g in R.groupby("block"):
            ax.scatter(g["nih_test_macro_auc"], g["vindr_test_macro_auc"], label=b, s=14)
        ax.set_xlabel("NIH test macro AUC")
        ax.set_ylabel("VinDr test macro AUC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_auc_internal_vs_external.png"), dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, a.smoke)
    outdir = os.path.join(cfg["paths"]["work_dir"], "analysis")
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(2026)
    nb, ci = cfg["analysis"]["n_boot"], cfg["analysis"]["ci"]

    R, A = load_runs(cfg)
    if R.empty:
        print("[analyze] nessun run valutato ancora.")
        return
    R.to_csv(os.path.join(outdir, "runs_table.csv"), index=False)
    mets = metric_cols(R)
    summ = R.groupby(["block", "condition"])[mets].agg(["mean", "std", "count"])
    summ.columns = [f"{m}__{s}" for m, s in summ.columns]
    summ.to_csv(os.path.join(outdir, "summary_by_condition.csv"))

    F = R[R["block"] == "factorial"].copy()
    eff, var = factorial_ols(F, mets) if len(F) >= 6 else (pd.DataFrame(), pd.DataFrame())
    eff.to_csv(os.path.join(outdir, "factorial_effects.csv"), index=False)
    var.to_csv(os.path.join(outdir, "factorial_variance.csv"), index=False)

    FA = F[F["run_id"].isin(A["run_id"].unique())] if not A.empty else F.iloc[0:0]
    hb = localization_hboot(FA, A, nb, ci, rng) if len(FA) >= 4 else pd.DataFrame()
    hb.to_csv(os.path.join(outdir, "localization_hboot.csv"), index=False)
    agr = attribution_agreement(cfg, FA, nb, ci, rng)
    agr.to_csv(os.path.join(outdir, "attribution_agreement.csv"), index=False)

    cal_rows = []
    ref = R[(R["crop"] == 0) & (R["schedule"] == "fx10") & (R["aug"] == 1)]
    for (arch, loss), g in ref.groupby(["arch", "loss"]):
        for ds in ["nih_test", "vindr_test"]:
            for m in ["raw", "ts", "platt"]:
                row = dict(arch=arch, loss=loss if arch == "convnext_base" else f"{loss} ({arch})",
                           dataset=ds, recal=m, n_seeds=len(g))
                for k in ["ece", "ece_adaptive", "brier", "nll", "slope", "intercept"]:
                    c = f"{ds}_{m}_{k}"
                    if c in g:
                        row[f"{k}_mean"], row[f"{k}_sd"] = g[c].mean(), g[c].std()
                row["auc_mean"] = g.get(f"{ds}_macro_auc", pd.Series(dtype=float)).mean()
                cal_rows.append(row)
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(os.path.join(outdir, "calibration_by_loss.csv"), index=False)

    fs_cols = [c for c in R.columns if c.startswith(("faith_", "sanity_"))]
    if fs_cols:
        R.groupby("condition")[fs_cols].agg(["mean", "std"]).to_csv(os.path.join(outdir, "faithfulness_sanity.csv"))

    try:
        figures(R, cal.rename(columns={}), agr, outdir)
    except Exception as e:
        print(f"[analyze] figure non generate: {e}")

    plan = build_plan(cfg)
    lines = ["# Report automatico (versione A)", "",
             f"Run valutati: {len(R)} / {len(plan)} pianificati; con attribuzione: {A['run_id'].nunique() if not A.empty else 0}", ""]
    for b, n in R.groupby("block").size().items():
        lines.append(f"- blocco {b}: {n} run completati")
    show = [c for c in ["nih_test_macro_auc", "vindr_test_macro_auc", "nih_test_raw_ece", "nih_test_ts_ece",
                        "nih_test_platt_ece", "vindr_test_raw_ece", "vindr_test_platt_ece"] if c in R]
    lines += ["", "## Medie per condizione (mean ± SD tra seed)", ""]
    for (b, c), g in R.groupby(["block", "condition"]):
        vals = ", ".join(f"{k}={g[k].mean():.4f}±{(g[k].std() if len(g) > 1 else 0):.4f}" for k in show)
        lines.append(f"- **{c}** ({b}, n={len(g)}): {vals}")
    if not eff.empty:
        lines += ["", "## Effetti principali (fattoriale, OLS a livello di run)", ""]
        for _, r in eff[eff["term"].isin(FACTORS)].iterrows():
            if r["metric"] in show or r["metric"].startswith("loc_vindr_gradcam"):
                lines.append(f"- {r['metric']} ~ {r['term']}: {r['estimate']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] "
                             f"(p={r['p']:.3g}, SD tra seed={r['resid_sd_seed']:.4f}, n={r['n_runs']})")
    # --- variabilita' tra seed: interna vs esterna ---------------------------------
    if len(F) and "vindr_test_macro_auc" in F:
        sd = F.groupby("condition")[[c for c in ["nih_test_macro_auc", "vindr_test_macro_auc",
                                                  "nih_test_matched_auc", "vindr_test_matched_auc"] if c in F]].std()
        lines += ["", "## Variabilita' tra seed (SD entro cella, media sulle celle del fattoriale)", ""]
        for c in sd.columns:
            lines.append(f"- {c}: SD media = {sd[c].mean():.4f} (min {sd[c].min():.4f}, max {sd[c].max():.4f})")
    if "nih_test_matched_auc" in R and "vindr_test_matched_auc" in R:
        lines += ["", "## AUC sulle classi comuni (8 classi), media ± SD su tutti i run per blocco/arch", ""]
        for (b, a_), g in R.groupby(["block", "arch"]):
            lines.append(f"- {b}/{a_} (n={len(g)}): NIH {g['nih_test_matched_auc'].mean():.4f}±{g['nih_test_matched_auc'].std():.4f} | "
                         f"VinDr {g['vindr_test_matched_auc'].mean():.4f}±{g['vindr_test_matched_auc'].std():.4f}")
    if not hb.empty:
        lines += ["", "## Effetti sulla localizzazione (bootstrap gerarchico seed x immagini)", ""]
        for _, r in hb.iterrows():
            lines.append(f"- {r['dataset']}/{r['method']}/{r['metric']} ~ {r['factor']}: {r['effect']:+.4f} "
                         f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]{' *' if r['ci_excludes_zero'] else ''}")
    fs = [c for c in R.columns if c.startswith(("faith_vindr_", "sanity_vindr_", "loc_"))]
    if fs:
        lines += ["", "## Attribuzione: medie su tutti i run (± SD tra run)", ""]
        for c in sorted(fs):
            lines.append(f"- {c}: {R[c].mean():.4f} ± {R[c].std():.4f} (n={R[c].notna().sum()})")
    if not agr.empty:
        lines += ["", "## Accordo attribuzioni (Spearman medio, IC bootstrap)", ""]
        for _, r in agr.iterrows():
            lines.append(f"- {r['dataset']} / {r['method']} / {r['comparison']}: {r['mean_spearman']:.3f} "
                         f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]")
    lines += ["", "_Generato da 06_analyze.py: numeri da verificare sulle tabelle CSV prima di riportarli nel paper._"]
    with open(os.path.join(outdir, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[analyze] {len(R)} run analizzati -> {outdir}")


if __name__ == "__main__":
    main()
