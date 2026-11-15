"""
08_extension_analysis.py -- analisi dell'ESTENSIONE 2026-11 (vedi PREREGISTRATION.md).

    python 08_extension_analysis.py [--config config.yaml]

Gruppi confrontati (cella di riferimento: crop=0, aug=1, cosine 10 epoche; 5 seed x {BCE, ASL}):
  convnext   ConvNeXt-Base (blocco factorial, cella di riferimento)   -- gia' nel paper
  losses     ConvNeXt-Base wBCE/focal (blocco losses)                -- gia' nel paper (solo cascata, IG-blur, descrittive)
  densenet   DenseNet-121 (blocco arch)                               -- gia' nel paper
  swin       Swin-B (blocco swin)                                     -- NUOVO
  regime     ConvNeXt-Base, regime alternativo wc15 + aug forte       -- NUOVO

Ipotesi pre-specificate (H1-H5, famiglia corretta con Holm; vedi PREREGISTRATION.md):
  H1 swin, H1 regime : rapporto SD esterna/interna (AUC sulle 8 classi comuni) > 1
  H2 swin, H2 regime : con ASL, ECE interna dopo temperature scaling > ECE dopo Platt (appaiato per seed)
  H3 swin            : effetto ASL vs BCE su energy-in-box VinDr con segno opposto fra Grad-CAM e Grad-CAM++
                       (test: differenza fra i due effetti != 0, appaiato per seed)
  H4                 : dopo la randomizzazione a cascata COMPLETA, rho(IG nero) > rho(Grad-CAM) (appaiato per run)
  H5 regime          : epoca selezionata nel regime alternativo > epoca selezionata nella ricetta del paper
Tutto il resto e' descrittivo (stime con IC al 95%).

Output in <work_dir>/analysis_ext/: ext_runs.csv, ext_*.csv, EXT_REPORT.md
"""
import os
import json
import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import load_config, CLASSES
from experiments import build_plan

MATCHED = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Pneumothorax", "Consolidation",
           "Fibrosis", "Pleural_Thickening"]
GROUP_LABEL = {"convnext": "ConvNeXt-Base (paper recipe)", "densenet": "DenseNet-121 (paper recipe)",
               "swin": "Swin-B (paper recipe)", "regime": "ConvNeXt-Base (alternative regime)",
               "losses": "ConvNeXt-Base, wBCE/focal (paper recipe)"}


def group_of(r):
    if r["block"] == "swin":
        return "swin"
    if r["block"] == "regime":
        return "regime"
    if r["block"] == "arch":
        return "densenet"
    if r["block"] == "losses":
        return "losses"
    if r["block"] == "factorial" and r["crop"] == 0 and r["schedule"] == "fx10" and r["aug"] == 1:
        return "convnext"
    return None


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def load(cfg):
    rows, maps = [], {}
    wd = cfg["paths"]["work_dir"]
    for r in build_plan(cfg):
        g = group_of(r)
        if g is None:
            continue
        rd = os.path.join(wd, "runs", r["run_id"])
        ev = jload(os.path.join(rd, "eval.json"))
        if ev is None:
            continue
        row = dict(r, group=g)
        row.update({k: v for k, v in ev.items() if k not in r})
        pc_p = os.path.join(rd, "eval_perclass.csv")
        if os.path.exists(pc_p):
            pc = pd.read_csv(pc_p)
            for ds in ["nih_test", "vindr_test"]:
                sub = pc[(pc["dataset"] == ds) & (pc["target"].isin(MATCHED))]
                row[f"{ds}_matched_auc"] = float(sub["auc"].mean()) if len(sub) else np.nan
        td = jload(os.path.join(rd, "train_done.json"))
        if td:
            h = td["history"]
            row["best_epoch"] = int(h[int(np.argmax([e["val_macro_auc"] for e in h]))]["epoch"])
            row["epochs_run"] = td["epochs_run"]
            row["best_val_auc"] = td["best_val_macro_auc"]
            row["val_auc_drop_last"] = float(td["best_val_macro_auc"] - h[-1]["val_macro_auc"])
        ap = os.path.join(rd, "attr_pairs.csv")
        if os.path.exists(ap):
            a = pd.read_csv(ap)
            a = a[a["dataset"] == "vindr"]
            for m, v in a.groupby("method")[["pointing", "energy", "norm_energy"]].mean().iterrows():
                for k in v.index:
                    row[f"loc_{m}_{k}"] = v[k]
        fp = os.path.join(rd, "attr_faithfulness.csv")
        if os.path.exists(fp) and os.path.getsize(fp) > 5:
            f = pd.read_csv(fp)
            for m, v in f[f["dataset"] == "vindr"].groupby("method")["faith_gain"].mean().items():
                row[f"faith_{m}"] = v
        bp = os.path.join(rd, "attr_ext_igblur_pairs.csv")
        if os.path.exists(bp) and os.path.getsize(bp) > 5:
            b = pd.read_csv(bp)
            b = b[b["dataset"] == "vindr"]
            for k in ["pointing", "energy", "norm_energy"]:
                row[f"loc_ig_blur_{k}"] = b[k].mean()
        bf = os.path.join(rd, "attr_ext_igblur_faith.csv")
        if os.path.exists(bf) and os.path.getsize(bf) > 5:
            f = pd.read_csv(bf)
            row["faith_ig_blur"] = f[f["dataset"] == "vindr"]["faith_gain"].mean()
        cp = os.path.join(rd, "attr_ext_cascade.csv")
        if os.path.exists(cp) and os.path.getsize(cp) > 5:
            c = pd.read_csv(cp)
            full = c[c["level"] == c["level"].max()]
            top = c[c["level"] == 2]          # testa + ultimo stadio = test parziale del paper
            for m, v in full.groupby("method")["spearman"].mean().items():
                row[f"cascade_full_{m}"] = v
            for m, v in top.groupby("method")["spearman"].mean().items():
                row[f"cascade_top2_{m}"] = v
        rows.append(row)
        maps[r["run_id"]] = rd
    return pd.DataFrame(rows), maps


# ---------------------------------------------------------------------
def t_ci(x, ci=0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    q = stats.t.ppf(0.5 + ci / 2, len(x) - 1)
    return m, m - q * se, m + q * se, se


def paired_by_seed(df, col, a_mask, b_mask):
    """Differenze (a - b) appaiate per seed."""
    A = df[a_mask].set_index("seed")[col]
    B = df[b_mask].set_index("seed")[col]
    s = sorted(set(A.index) & set(B.index))
    return np.array([A[k] - B[k] for k in s], float), s


def sd_ratio_test(G):
    """Rapporto SD esterna/interna (matched AUC), SD entro cella (loss) mediata sulle celle;
    incertezza sui seed con jackknife leave-one-seed-out sul log-rapporto, t con n-1 gdl (come nel paper)."""
    def ratio(sub):
        sdi = sub.groupby("loss")["nih_test_matched_auc"].std().mean()
        sde = sub.groupby("loss")["vindr_test_matched_auc"].std().mean()
        return sdi, sde, sde / sdi if sdi > 0 else np.nan
    sdi, sde, R = ratio(G)
    seeds = sorted(G["seed"].unique())
    jk = np.array([np.log(ratio(G[G["seed"] != s])[2]) for s in seeds]) if len(seeds) >= 3 else np.array([])
    out = dict(sd_internal=sdi, sd_external=sde, ratio=R, n_seeds=len(seeds),
               loo_min=float(np.exp(jk.min())) if len(jk) else np.nan,
               loo_max=float(np.exp(jk.max())) if len(jk) else np.nan)
    if len(jk) >= 3 and np.isfinite(R) and R > 0:
        n = len(jk)
        se = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
        tcrit = stats.t.ppf(0.975, n - 1)
        out.update(ci_lo=float(np.exp(np.log(R) - tcrit * se)), ci_hi=float(np.exp(np.log(R) + tcrit * se)),
                   p_one_sided=float(stats.t.sf(np.log(R) / se, n - 1)) if se > 0 else np.nan)
    return out


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    adj = np.full(len(p), np.nan)
    running = 0.0
    m = int(np.isfinite(p).sum())
    for rank, i in enumerate(order):
        if not np.isfinite(p[i]):
            continue
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def seed_agreement(R, maps, group, loss, method, g_small=32):
    """Spearman medio fra mappe della STESSA condizione e seed diversi (coppie VinDr presenti in entrambe)."""
    from scipy.stats import spearmanr
    import torch
    import torch.nn.functional as F
    sub = R[(R["group"] == group) & (R["loss"] == loss)]
    key, fname = (("map_ig_blur", "attr_ext_maps.npz") if method == "ig_blur" else (f"map_{method}", "attr_maps.npz"))
    M = {}
    for rid in sub["run_id"]:
        p = os.path.join(maps[rid], fname)
        if not os.path.exists(p):
            continue
        d = np.load(p, allow_pickle=True)
        if key not in d:
            continue
        ids = d["pair_id"].astype(str)
        sel = np.array([i.startswith("vindr|") for i in ids])
        x = torch.from_numpy(d[key][sel].astype(np.float32))
        ok = ~torch.isnan(x.view(len(x), -1)).any(1)
        x = F.interpolate(torch.nan_to_num(x).unsqueeze(1), size=(g_small, g_small), mode="area")[:, 0]
        M[rid] = (ids[sel][ok.numpy()], x[ok].view(int(ok.sum()), -1).numpy())
    vals = []
    for r1, r2 in itertools.combinations(sorted(M), 2):
        i1, x1 = M[r1]
        i2, x2 = M[r2]
        common = sorted(set(i1) & set(i2))
        if not common:
            continue
        p1 = {k: j for j, k in enumerate(i1)}
        p2 = {k: j for j, k in enumerate(i2)}
        rhos = []
        for k in common:
            a, b = x1[p1[k]], x2[p2[k]]
            if a.std() > 0 and b.std() > 0:
                rhos.append(spearmanr(a, b).statistic)
        if rhos:
            vals.append(np.mean(rhos))
    return (float(np.mean(vals)), len(vals)) if vals else (np.nan, 0)


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = os.path.join(cfg["paths"]["work_dir"], "analysis_ext")
    os.makedirs(out, exist_ok=True)
    R, maps = load(cfg)
    if R.empty:
        print("[ext] nessun run valutato.")
        return
    R.to_csv(os.path.join(out, "ext_runs.csv"), index=False)
    L = ["# Estensione 2026-11: report automatico", "",
         "_Numeri da verificare sui CSV prima di riportarli nel paper._", ""]
    for g, n in R.groupby("group").size().items():
        L.append(f"- {GROUP_LABEL[g]}: {n} run valutati")

    # ---------------- discriminazione e rapporto SD -----------------------
    disc = []
    for g, G in R.groupby("group"):
        row = dict(group=g, n_runs=len(G),
                   nih_matched_auc=G["nih_test_matched_auc"].mean(), vindr_matched_auc=G["vindr_test_matched_auc"].mean())
        row.update(sd_ratio_test(G) if G["seed"].nunique() >= 2 else {})
        disc.append(row)
    D = pd.DataFrame(disc)
    D.to_csv(os.path.join(out, "ext_discrimination.csv"), index=False)
    L += ["", "## Discriminazione (classi comuni) e variabilita' tra seed", ""]
    for _, r in D.iterrows():
        L.append(f"- {GROUP_LABEL[r['group']]}: AUC interna {r['nih_matched_auc']:.3f}, esterna {r['vindr_matched_auc']:.3f}; "
                 f"SD tra seed {r.get('sd_internal', np.nan):.4f} vs {r.get('sd_external', np.nan):.4f}, "
                 f"rapporto {r.get('ratio', np.nan):.1f} [IC {r.get('ci_lo', np.nan):.1f}, {r.get('ci_hi', np.nan):.1f}], "
                 f"leave-one-seed-out {r.get('loo_min', np.nan):.1f}–{r.get('loo_max', np.nan):.1f}")

    # ---------------- epoca selezionata -----------------------------------
    if "best_epoch" in R:
        E = R.groupby(["group", "loss"]).agg(best_epoch_mean=("best_epoch", "mean"), best_epoch_min=("best_epoch", "min"),
                                             best_epoch_max=("best_epoch", "max"), val_drop=("val_auc_drop_last", "mean"),
                                             epochs_run=("epochs_run", "mean")).reset_index()
        E.to_csv(os.path.join(out, "ext_selected_epoch.csv"), index=False)
        L += ["", "## Epoca selezionata (miglior AUC di validazione)", ""]
        for _, r in E.iterrows():
            L.append(f"- {GROUP_LABEL[r['group']]} / {r['loss']}: media {r['best_epoch_mean']:.1f} "
                     f"(min {r['best_epoch_min']}, max {r['best_epoch_max']}) su {r['epochs_run']:.0f} epoche; "
                     f"calo AUC val dal picco all'ultima epoca {r['val_drop']:.3f}")

    # ---------------- calibrazione ----------------------------------------
    cal = []
    for (g, loss), G in R.groupby(["group", "loss"]):
        for ds in ["nih_test", "vindr_test"]:
            for m in ["raw", "ts", "platt"]:
                c = dict(group=g, loss=loss, dataset=ds, recal=m, n=len(G))
                for k in ["ece", "intercept", "slope"]:
                    col = f"{ds}_{m}_{k}"
                    if col in G:
                        c[f"{k}_mean"], c[f"{k}_sd"] = G[col].mean(), G[col].std()
                cal.append(c)
    C = pd.DataFrame(cal)
    C.to_csv(os.path.join(out, "ext_calibration.csv"), index=False)
    L += ["", "## Calibrazione (ECE con 10 bin pesati; media sui seed)", ""]
    for (g, loss), G in C.groupby(["group", "loss"]):
        get = lambda ds, m, k="ece": G[(G["dataset"] == ds) & (G["recal"] == m)][f"{k}_mean"].mean()
        L.append(f"- {GROUP_LABEL[g]} / {loss}: interna raw {get('nih_test', 'raw'):.3f}, TS {get('nih_test', 'ts'):.3f}, "
                 f"Platt {get('nih_test', 'platt'):.3f} (intercetta raw {get('nih_test', 'raw', 'intercept'):+.2f}); "
                 f"esterna raw {get('vindr_test', 'raw'):.3f}, TS {get('vindr_test', 'ts'):.3f}, Platt {get('vindr_test', 'platt'):.3f}")

    # ---------------- localizzazione: effetto ASL vs BCE per metodo ---------
    loc = []
    for g, G in R.groupby("group"):
        for meth in ["gradcam", "gradcampp", "ig", "ig_blur"]:
            for k in ["pointing", "energy", "norm_energy"]:
                col = f"loc_{meth}_{k}"
                if col not in G:
                    continue
                d, s = paired_by_seed(G, col, G["loss"] == "asl", G["loss"] == "bce")
                m, lo, hi, _ = t_ci(d)
                loc.append(dict(group=g, method=meth, metric=k, bce_mean=G[G["loss"] == "bce"][col].mean(),
                                asl_mean=G[G["loss"] == "asl"][col].mean(), asl_minus_bce=m, ci_lo=lo, ci_hi=hi,
                                n_seeds=len(s)))
    LOC = pd.DataFrame(loc)
    LOC.to_csv(os.path.join(out, "ext_localization.csv"), index=False)
    if len(LOC):
        L += ["", "## Localizzazione VinDr: effetto ASL - BCE (appaiato per seed, IC t al 95%)", ""]
        for _, r in LOC[LOC["metric"].isin(["energy", "norm_energy"])].iterrows():
            L.append(f"- {GROUP_LABEL[r['group']]} / {r['method']} / {r['metric']}: {r['asl_minus_bce']:+.4f} "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] (BCE {r['bce_mean']:.3f}, ASL {r['asl_mean']:.3f})")

    # ---------------- accordo fra seed ----------------------------------
    agr = []
    for (g, loss) in R.groupby(["group", "loss"]).size().index:
        for meth in ["gradcam", "gradcampp", "ig", "ig_blur"]:
            v, n = seed_agreement(R, maps, g, loss, meth)
            agr.append(dict(group=g, loss=loss, method=meth, mean_spearman_between_seeds=v, n_seed_pairs=n))
    AG = pd.DataFrame(agr)
    AG.to_csv(os.path.join(out, "ext_seed_agreement.csv"), index=False)
    L += ["", "## Accordo fra mappe di seed diversi (stessa condizione, VinDr)", ""]
    for (g, meth), G in AG.groupby(["group", "method"]):
        L.append(f"- {GROUP_LABEL[g]} / {meth}: rho medio {G['mean_spearman_between_seeds'].mean():.3f}")

    # ---------------- cascata e IG-blur ----------------------------------
    casc_files = [os.path.join(maps[r], "attr_ext_cascade.csv") for r in R["run_id"]]
    cc = [pd.read_csv(f).assign(group=R.set_index("run_id").loc[os.path.basename(os.path.dirname(f)), "group"])
          for f in casc_files if os.path.exists(f) and os.path.getsize(f) > 5]
    if cc:
        CC = pd.concat(cc, ignore_index=True)
        CS = CC.groupby(["group", "method", "level", "level_name"])["spearman"].agg(["mean", "std", "count"]).reset_index()
        CS.to_csv(os.path.join(out, "ext_cascade_curves.csv"), index=False)
        L += ["", "## Randomizzazione a cascata: rho originale vs randomizzato dopo l'ultimo livello (VinDr)", ""]
        full = CC[CC["level"] == CC.groupby("run_id")["level"].transform("max")]
        for (g, meth), G in full.groupby(["group", "method"]):
            L.append(f"- {GROUP_LABEL[g]} / {meth}: {G['spearman'].mean():.3f} ± {G['spearman'].std():.3f}")
    blur_cols = [c for c in R.columns if "ig_blur" in c]
    if blur_cols:
        L += ["", "## IG con baseline nera vs sfocata (VinDr, media sui run)", ""]
        for k in ["loc_{}_energy", "loc_{}_norm_energy", "faith_{}", "cascade_full_{}"]:
            a_, b_ = k.format("ig"), k.format("ig_blur")
            if k.startswith("cascade"):
                a_ = "cascade_full_ig_black"
            if a_ in R and b_ in R:
                L.append(f"- {k.format('*')}: nera {R[a_].mean():.3f}, sfocata {R[b_].mean():.3f}")

    # ---------------- ipotesi pre-specificate ----------------------------
    H = []
    for g in ["swin", "regime"]:
        r = D[D["group"] == g]
        if len(r):
            H.append(dict(h=f"H1 {g}", desc="rapporto SD esterna/interna > 1", estimate=r["ratio"].iloc[0],
                          ci=f"[{r.get('ci_lo', pd.Series([np.nan])).iloc[0]:.2f}, {r.get('ci_hi', pd.Series([np.nan])).iloc[0]:.2f}]",
                          p=r.get("p_one_sided", pd.Series([np.nan])).iloc[0]))
        G = R[(R["group"] == g) & (R["loss"] == "asl")]
        if len(G) >= 3 and "nih_test_ts_ece" in G:
            d = (G["nih_test_ts_ece"] - G["nih_test_platt_ece"]).values
            m, lo, hi, se = t_ci(d)
            p = float(stats.t.sf(m / se, len(d) - 1)) if se and se > 0 else np.nan
            H.append(dict(h=f"H2 {g}", desc="ASL: ECE interna TS - Platt > 0", estimate=m, ci=f"[{lo:.4f}, {hi:.4f}]", p=p))
    G = R[R["group"] == "swin"]
    if len(G) and "loc_gradcam_energy" in G and "loc_gradcampp_energy" in G:
        d1, s1 = paired_by_seed(G, "loc_gradcam_energy", G["loss"] == "asl", G["loss"] == "bce")
        d2, s2 = paired_by_seed(G, "loc_gradcampp_energy", G["loss"] == "asl", G["loss"] == "bce")
        if len(d1) >= 3 and s1 == s2:
            d = d1 - d2
            m, lo, hi, se = t_ci(d)
            p = float(2 * stats.t.sf(abs(m) / se, len(d) - 1)) if se > 0 else np.nan
            H.append(dict(h="H3 swin", desc="effetto ASL su energy-in-box: Grad-CAM - Grad-CAM++ != 0 "
                          f"(Grad-CAM {np.mean(d1):+.4f}, Grad-CAM++ {np.mean(d2):+.4f})",
                          estimate=m, ci=f"[{lo:.4f}, {hi:.4f}]", p=p))
    if "cascade_full_ig_black" in R and "cascade_full_gradcam" in R:
        d = (R["cascade_full_ig_black"] - R["cascade_full_gradcam"]).dropna().values
        if len(d) >= 3:
            m, lo, hi, se = t_ci(d)
            p = float(stats.t.sf(m / se, len(d) - 1)) if se > 0 else np.nan
            H.append(dict(h="H4", desc="cascata completa: rho(IG nero) - rho(Grad-CAM) > 0 (tutti i run)",
                          estimate=m, ci=f"[{lo:.3f}, {hi:.3f}]", p=p))
    if "best_epoch" in R:
        a1 = R[R["group"] == "regime"]["best_epoch"].values
        a0 = R[R["group"] == "convnext"]["best_epoch"].values
        if len(a1) >= 3 and len(a0) >= 3:
            res = stats.mannwhitneyu(a1, a0, alternative="greater")
            H.append(dict(h="H5 regime", desc="epoca selezionata: regime alternativo > ricetta del paper (Mann-Whitney)",
                          estimate=float(np.median(a1) - np.median(a0)), ci="", p=float(res.pvalue)))
    HT = pd.DataFrame(H)
    if len(HT):
        HT["p_holm"] = holm(HT["p"].values)
        HT.to_csv(os.path.join(out, "ext_hypotheses.csv"), index=False)
        L += ["", "## Ipotesi pre-specificate (famiglia H1-H5, correzione di Holm)", ""]
        for _, r in HT.iterrows():
            L.append(f"- **{r['h']}** {r['desc']}: stima {r['estimate']:.4g} {r['ci']}; p = {r['p']:.3g}; "
                     f"p Holm = {r['p_holm']:.3g}")
    open(os.path.join(out, "EXT_REPORT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
