"""
09_demographics.py -- caratteristiche demografiche dei dati usati (eta' e sesso), per CLAIM item 36.

    python 09_demographics.py                 # NIH + VinDr
    python 09_demographics.py --skip-vindr    # solo NIH (nessun accesso di rete)

NIH ChestX-ray14: eta', sesso e proiezione da Data_Entry_2017.csv (in paths.nih_dir), uniti ai manifest
  della pipeline (nih_train/val/test.csv). Riportati per immagine e per paziente (prima immagine del paziente).
  Eta' > 100 anni: valori noti come errati nel file NIH -> trattati come mancanti e contati.

VinDr-CXR (release Kaggle): la cache 512 px non conserva i tag DICOM e i DICOM non sono su disco
  (01b li legge in streaming). Questo script rilegge dallo zip remoto SOLO l'intestazione di ogni DICOM
  (primi 256 KB del membro, stop_before_pixels): ~15.000 x 0.25 MB = ~4 GB di traffico, rilanciabile.
  Tag usati: PatientAge (es. "045Y") e PatientSex. Molti valori possono essere vuoti (anonimizzazione):
  vengono contati come mancanti, non imputati.

Output in <work_dir>/analysis_ext/: demographics.csv, demographics.md
Cache intermedia VinDr: <work_dir>/data/vindr_dicom_header_tags.csv
"""
import io
import os
import re
import time
import argparse
import importlib
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from common import load_config, shared_data_dir, load_manifest


def summarize(df, label, age_col="age", sex_col="sex", per="image"):
    n = len(df)
    ages = pd.to_numeric(df[age_col], errors="coerce")
    valid = ages.notna()
    sex = df[sex_col].astype(str).str.upper().str[:1].where(df[sex_col].notna(), "")
    known = sex.isin(["M", "F"])
    return dict(dataset=label, unit=per, n=n,
                age_known=int(valid.sum()),
                age_mean=float(ages[valid].mean()) if valid.any() else np.nan,
                age_sd=float(ages[valid].std()) if valid.sum() > 1 else np.nan,
                age_median=float(ages[valid].median()) if valid.any() else np.nan,
                age_q1=float(ages[valid].quantile(0.25)) if valid.any() else np.nan,
                age_q3=float(ages[valid].quantile(0.75)) if valid.any() else np.nan,
                age_min=float(ages[valid].min()) if valid.any() else np.nan,
                age_max=float(ages[valid].max()) if valid.any() else np.nan,
                age_under18=int((ages[valid] < 18).sum()),
                sex_known=int(known.sum()), female=int((sex == "F").sum()), male=int((sex == "M").sum()),
                female_pct_of_known=float((sex == "F").sum() / known.sum() * 100) if known.any() else np.nan)


# ------------------------------- NIH ---------------------------------
def nih_demographics(cfg):
    src = os.path.join(cfg["paths"]["nih_dir"], "Data_Entry_2017.csv")
    if os.path.exists(src):
        meta = pd.read_csv(src)
    else:   # i file di split contengono le stesse colonne di Data_Entry_2017.csv
        meta = pd.concat([pd.read_csv(os.path.join(cfg["paths"]["splits_dir"], f"{s}_split.csv"))
                          for s in ["train", "val", "test"]], ignore_index=True)
    meta = meta.rename(columns={"Image Index": "image_id", "Patient ID": "pid", "Patient Age": "age_raw",
                                "Patient Gender": "sex", "View Position": "view"})
    meta["age"] = pd.to_numeric(meta["age_raw"], errors="coerce")
    n_bad = int((meta["age"] > 100).sum())
    meta.loc[meta["age"] > 100, "age"] = np.nan
    out, notes = [], [f"NIH: {n_bad} immagini con eta' > 100 anni nel file originale, trattate come mancanti."]
    dd = shared_data_dir(cfg)
    for split in ["train", "val", "test"]:
        m = load_manifest(cfg, f"nih_{split}")[["image_id"]].merge(meta, on="image_id", how="left")
        r = summarize(m, f"NIH {split}", per="image")
        r["pa_view_pct"] = float((m["view"] == "PA").mean() * 100)
        r["patients"] = int(m["pid"].nunique())
        out.append(r)
        first = m.sort_values(["pid", "image_id"]).groupby("pid").head(1)
        rp = summarize(first, f"NIH {split}", per="patient (first image)")
        rp["patients"] = int(m["pid"].nunique())
        out.append(rp)
    bb = os.path.join(dd, "nih_bbox_test.csv")
    if os.path.exists(bb):
        ids = pd.read_csv(bb)["image_id"].astype(str).unique()
        m = meta[meta["image_id"].isin(ids)]
        r = summarize(m, "NIH test, images with bounding boxes", per="image")
        r["pa_view_pct"] = float((m["view"] == "PA").mean() * 100)
        out.append(r)
    return out, notes


# ------------------------------- VinDr -------------------------------
def parse_age(s):
    if s is None:
        return np.nan
    m = re.match(r"^\s*(\d{1,3})\s*([YMWD]?)\s*$", str(s).upper())
    if not m:
        return np.nan
    v, u = float(m.group(1)), m.group(2) or "Y"
    if v == 0:                      # 000Y / 000D: segnaposto della release, non eta' reali
        return np.nan
    a = v if u == "Y" else v / 12 if u == "M" else v / 52.1775 if u == "W" else v / 365.25
    return a if a <= 100 else np.nan  # come per NIH: > 100 anni trattato come mancante


def vindr_tags(cfg, threads):
    dl = importlib.import_module("01b_download_vindr_kaggle_stream")
    import pydicom
    cache = os.path.join(shared_data_dir(cfg), "vindr_dicom_header_tags.csv")
    ids = load_manifest(cfg, "vindr_test")["image_id"].astype(str).tolist()
    have = pd.read_csv(cache, dtype=str) if os.path.exists(cache) else pd.DataFrame(columns=["image_id"])
    todo = sorted(set(ids) - set(have["image_id"]))
    print(f"[VinDr] intestazioni gia' lette: {len(ids) - len(todo)} | da leggere: {len(todo)}")
    if todo:
        urls = dl.UrlProvider(dl.kaggle_zip_url)
        rf = dl.RemoteFile(urls, block=256 << 10)
        infos = {i.filename: i for i in zipfile.ZipFile(rf).infolist()}
        local, lock, rows = threading.local(), threading.Lock(), []

        def one(iid):
            try:
                if not hasattr(local, "zf"):
                    local.zf = zipfile.ZipFile(dl.RemoteFile(urls, size=rf.size, block=256 << 10))
                info = infos[f"train/{iid}.dicom"]
                for nbytes in (256 << 10, 2 << 20):
                    with local.zf.open(info) as fh:
                        head = fh.read(nbytes)
                    try:
                        ds = pydicom.dcmread(io.BytesIO(head), stop_before_pixels=True, force=True)
                        return dict(image_id=iid, PatientAge=str(ds.get("PatientAge", "") or ""),
                                    PatientSex=str(ds.get("PatientSex", "") or ""), error="")
                    except Exception as e:      # intestazione piu' lunga del blocco letto: riprova con 2 MB
                        last = e
                return dict(image_id=iid, PatientAge="", PatientSex="", error=f"{type(last).__name__}: {last}")
            except Exception as e:
                return dict(image_id=iid, PatientAge="", PatientSex="", error=f"{type(e).__name__}: {e}")

        t0 = time.time()
        with ThreadPoolExecutor(threads) as ex:
            futs = [ex.submit(one, i) for i in todo]
            for k, f in enumerate(as_completed(futs), 1):
                with lock:
                    rows.append(f.result())
                if k % 500 == 0 or k == len(todo):
                    print(f"[VinDr] {k}/{len(todo)} ({k / max(time.time() - t0, 1e-6):.1f} img/s)", flush=True)
                    pd.concat([have, pd.DataFrame(rows)], ignore_index=True).to_csv(cache + ".tmp", index=False)
                    os.replace(cache + ".tmp", cache)
        have = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
        have.to_csv(cache + ".tmp", index=False)
        os.replace(cache + ".tmp", cache)
    return have[have["image_id"].isin(ids)].copy()


def vindr_demographics(cfg, threads):
    t = vindr_tags(cfg, threads)
    t["age"] = t["PatientAge"].map(parse_age)
    t["sex"] = t["PatientSex"].where(t["PatientSex"].astype(str).str.upper().str[:1].isin(["M", "F"]), np.nan)
    n_err = int((t["error"].fillna("") != "").sum())
    out = [summarize(t, "VinDr-CXR external test (15 000)", per="image")]
    dd = shared_data_dir(cfg)
    bb = os.path.join(dd, "vindr_bbox_test.csv")
    if os.path.exists(bb):
        ids = pd.read_csv(bb)["image_id"].astype(str).unique()
        out.append(summarize(t[t["image_id"].isin(ids)], "VinDr-CXR, images with bounding boxes", per="image"))
    notes = [f"VinDr: intestazioni non leggibili: {n_err}; eta' mancante o non interpretabile: "
             f"{int(t['age'].isna().sum())}; sesso mancante o diverso da M/F: {int(t['sex'].isna().sum())}.",
             "VinDr: l'identificativo paziente non e' disponibile nella release, quindi i conteggi sono per immagine."]
    return out, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--skip-vindr", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    cfg = load_config(a.config)
    outdir = os.path.join(cfg["paths"]["work_dir"], "analysis_ext")
    os.makedirs(outdir, exist_ok=True)
    rows, notes = nih_demographics(cfg)
    if not a.skip_vindr:
        r2, n2 = vindr_demographics(cfg, a.threads)
        rows += r2
        notes += n2
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "demographics.csv"), index=False)
    lines = ["# Caratteristiche demografiche", "",
             "| Dataset | Unita' | n | Eta' nota | Eta' media ± SD | Mediana [IQR] | <18 anni | Sesso noto | Donne (% dei noti) |",
             "|---|---|---:|---:|---|---|---:|---:|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['dataset']} | {r['unit']} | {r['n']} | {r['age_known']} | "
                     f"{r['age_mean']:.1f} ± {r['age_sd']:.1f} | {r['age_median']:.0f} [{r['age_q1']:.0f}–{r['age_q3']:.0f}] | "
                     f"{r['age_under18']} | {r['sex_known']} | {r['female']} ({r['female_pct_of_known']:.1f}%) |")
    lines += ["", "Note:"] + [f"- {n}" for n in notes]
    open(os.path.join(outdir, "demographics.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
