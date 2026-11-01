"""
02_prepare_data.py -- crea la cache di immagini pre-ridimensionate e i manifest.

Perche' una cache: decodificare 112k PNG 1024x1024 a ogni epoca e' il collo di
bottiglia del training. Ridimensionarli UNA volta a 512x512 rende ogni run
molto piu' veloce. (Nota per il paper: la catena diventa 1024 -> 512 -> 384
invece di 1024 -> 384; va dichiarato nei metodi.)

Output in <work_dir>/data/:
  nih/*.png, vindr/*.png                   immagini 512x512, 8 bit, grayscale
  nih_train.csv, nih_val.csv, nih_test.csv  image_id, cache_path, patient_id, 14 colonne label
  nih_bbox_test.csv                         box ufficiali NIH ristretti al TEST split (coord. cache)
  vindr_test.csv                            3000 immagini test VinDr + etichette mappate (colonne vd_*)
  vindr_bbox_test.csv                       box di consenso VinDr test per i target mappati (coord. cache)
  prep_report.json                          conteggi e controlli

Rilanciabile: le immagini gia' in cache vengono saltate.
"""
import os
import glob
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import (load_config, shared_data_dir, vindr_dir, CLASSES, NIH_BBOX_TO_CLASS,
                    VINDR_TARGETS, norm_name, labels_from_string, load_pil_gray_any, to_uint8, dump_json)

CACHE = None
GEOM = None


# ------------------------------- NIH ---------------------------------
def _nih_worker(args):
    src, dst, CACHE = args
    with Image.open(src) as im:
        w, h = im.size
    if not os.path.exists(dst):
        img = load_pil_gray_any(src).resize((CACHE, CACHE), Image.BILINEAR, reducing_gap=2.0)
        tmp = dst + ".tmp.png"
        img.save(tmp)
        os.replace(tmp, dst)
    return os.path.basename(src), w, h


def prepare_nih(cfg, out_dir, workers):
    nih = cfg["paths"]["nih_dir"]
    files = glob.glob(os.path.join(nih, "images_*", "images", "*.png"))
    if not files:
        files = glob.glob(os.path.join(nih, "**", "*.png"), recursive=True)
    by_name = {os.path.basename(f): f for f in files}
    print(f"[NIH] PNG trovati: {len(by_name)}")

    splits = {}
    for s in ["train", "val", "test"]:
        df = pd.read_csv(os.path.join(cfg["paths"]["splits_dir"], f"{s}_split.csv"))
        splits[s] = df
    # controllo leakage paziente
    pid = {s: set(df["Patient ID"].astype(str)) for s, df in splits.items()}
    assert not (pid["train"] & pid["val"]) and not (pid["train"] & pid["test"]) and not (pid["val"] & pid["test"]), \
        "overlap di pazienti tra split!"

    cdir = os.path.join(out_dir, "nih")
    os.makedirs(cdir, exist_ok=True)
    needed = pd.concat(splits.values())["Image Index"].tolist()
    missing = [n for n in needed if n not in by_name]
    if missing:
        print(f"[NIH][ATTENZIONE] {len(missing)} immagini degli split non trovate su disco (es. {missing[:3]}); verranno escluse.")
    jobs = [(by_name[n], os.path.join(cdir, n), CACHE) for n in needed if n in by_name]
    sizes = {}
    with Pool(workers) as pool:
        for name, w, h in tqdm(pool.imap_unordered(_nih_worker, jobs, chunksize=64), total=len(jobs), desc="cache NIH"):
            sizes[name] = (w, h)

    report = {}
    for s, df in splits.items():
        df = df[df["Image Index"].isin(sizes)].copy()
        lab = np.stack([labels_from_string(x) for x in df["Finding Labels"]]) if len(df) else np.zeros((0, len(CLASSES)))
        out = pd.DataFrame({"image_id": df["Image Index"].values,
                            "cache_path": [os.path.join(cdir, n) for n in df["Image Index"]],
                            "patient_id": df["Patient ID"].astype(str).values,
                            "orig_w": [sizes[n][0] for n in df["Image Index"]],
                            "orig_h": [sizes[n][1] for n in df["Image Index"]]})
        for i, c in enumerate(CLASSES):
            out[c] = lab[:, i].astype(int)
        out.to_csv(os.path.join(out_dir, f"nih_{s}.csv"), index=False)
        report[f"nih_{s}_images"] = len(out)
        report[f"nih_{s}_patients"] = int(out["patient_id"].nunique())
        print(f"[NIH] {s}: {len(out)} immagini, {out['patient_id'].nunique()} pazienti")

    # bounding box ufficiali ristretti al test split
    bb = pd.read_csv(os.path.join(nih, "BBox_List_2017.csv"))
    coord = [c for c in bb.columns if c.lower() not in ("image index", "finding label") and not c.startswith("Unnamed")][:4]
    bb = bb.rename(columns={"Image Index": "image_id", "Finding Label": "label",
                            coord[0]: "x", coord[1]: "y", coord[2]: "w", coord[3]: "h"})
    bb["target"] = bb["label"].str.strip().map(NIH_BBOX_TO_CLASS)
    n_all = len(bb)
    test_ids = set(splits["test"]["Image Index"]) & set(sizes)
    bb = bb[bb["image_id"].isin(test_ids) & bb["target"].notna()].copy()
    sx = np.array([CACHE / sizes[n][0] for n in bb["image_id"]])
    sy = np.array([CACHE / sizes[n][1] for n in bb["image_id"]])
    out = pd.DataFrame({"image_id": bb["image_id"].values, "target": bb["target"].values,
                        "x1": bb["x"].values * sx, "y1": bb["y"].values * sy,
                        "x2": (bb["x"].values + bb["w"].values) * sx, "y2": (bb["y"].values + bb["h"].values) * sy})
    out.to_csv(os.path.join(out_dir, "nih_bbox_test.csv"), index=False)
    report.update(nih_bbox_all=n_all, nih_bbox_test=len(out), nih_bbox_test_images=int(out["image_id"].nunique()))
    print(f"[NIH] box: {n_all} totali -> {len(out)} nel test split ({out['image_id'].nunique()} immagini)")
    return report


# ------------------------------ VinDr --------------------------------
def read_dicom_uint8(path):
    import pydicom
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    try:
        from pydicom.pixels import apply_voi_lut          # pydicom >= 3
    except Exception:
        from pydicom.pixel_data_handlers.util import apply_voi_lut
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    arr = arr.astype(np.float64)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return to_uint8(arr)


def _vindr_worker(args):
    src, dst, CACHE, GEOM = args
    if os.path.exists(dst) and os.path.exists(dst + ".geom.npy"):
        g = np.load(dst + ".geom.npy")
        return os.path.splitext(os.path.basename(src))[0], *g.tolist()
    a = read_dicom_uint8(src)
    h, w = a.shape
    img = Image.fromarray(a)
    if GEOM == "pad":
        s = CACHE / max(w, h)
        nw, nh = int(round(w * s)), int(round(h * s))
        canvas = Image.new("L", (CACHE, CACHE), 0)
        ox, oy = (CACHE - nw) // 2, (CACHE - nh) // 2
        canvas.paste(img.resize((nw, nh), Image.BILINEAR, reducing_gap=2.0), (ox, oy))
        img, sx, sy = canvas, s, s
    else:
        img = img.resize((CACHE, CACHE), Image.BILINEAR, reducing_gap=2.0)
        sx, sy, ox, oy = CACHE / w, CACHE / h, 0, 0
    tmp = dst + ".tmp.png"
    img.save(tmp)
    os.replace(tmp, dst)
    g = np.array([w, h, sx, sy, ox, oy], dtype=np.float64)
    np.save(dst + ".geom.npy", g)
    return os.path.splitext(os.path.basename(src))[0], *g.tolist()


def find_ci(root, name):
    for p in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.basename(p).lower() == name.lower():
            return p
    raise FileNotFoundError(f"{name} non trovato sotto {root}")


def _cache_dicoms(dcm, out_dir, workers):
    cdir = os.path.join(out_dir, "vindr")
    os.makedirs(cdir, exist_ok=True)
    jobs = [(p, os.path.join(cdir, os.path.splitext(os.path.basename(p))[0] + ".png"), CACHE, GEOM) for p in dcm]
    geo = {}
    with Pool(workers) as pool:
        for r in tqdm(pool.imap_unordered(_vindr_worker, jobs, chunksize=8), total=len(jobs), desc="cache VinDr"):
            geo[r[0]] = r[1:]
    return cdir, geo


def prepare_vindr_kaggle(cfg, out_dir, workers):
    """Competizione Kaggle VinBigData: si usa lo split 'train' (15.000 immagini, 3 radiologi
    indipendenti per immagine). Non viene MAI usato per allenare: e' solo valutazione esterna.
    Etichette d'immagine: maggioranza (>=2/3) o 'any' (>=1/3), da config.
    Maschera di localizzazione: unione dei box di TUTTI i radiologi per quella classe.
    Classi presenti solo come etichette globali nella release PhysioNet (es. Pneumonia) qui
    non esistono e vengono saltate automaticamente."""
    root = cfg["paths"]["vindr_kaggle_dir"]
    dcm = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
           if p.lower().endswith((".dicom", ".dcm")) and os.path.basename(os.path.dirname(p)).lower() == "train"]
    print(f"[VinDr-Kaggle] DICOM train trovati: {len(dcm)}")
    cdir = os.path.join(out_dir, "vindr")
    # immagini gia' convertite (es. dal downloader a flusso 01b, che cancella i DICOM)
    geo = {}
    for g in glob.glob(os.path.join(cdir, "*.png.geom.npy")):
        iid = os.path.basename(g)[:-len(".png.geom.npy")]
        if os.path.exists(os.path.join(cdir, iid + ".png")):
            geo[iid] = tuple(np.load(g).tolist())
    print(f"[VinDr-Kaggle] immagini gia' in cache: {len(geo)}")
    if not dcm and not geo:
        print("[VinDr-Kaggle][ATTENZIONE] nessun DICOM e nessuna immagine in cache: salto la valutazione esterna.")
        return {"vindr_test_images": 0}
    csv = find_ci(root, "train.csv")
    if dcm:
        cdir, g2 = _cache_dicoms(dcm, out_dir, workers)
        geo.update(g2)
    ann = pd.read_csv(csv)
    ann["image_id"] = ann["image_id"].astype(str)
    ann = ann[ann["image_id"].isin(geo)].copy()
    name2tgt = {norm_name(v): t for t, (v, _) in VINDR_TARGETS.items()}
    ann["target"] = ann["class_name"].map(lambda x: name2tgt.get(norm_name(x)))
    found = sorted(ann["target"].dropna().unique())
    absent = [t for t in VINDR_TARGETS if t not in found]
    print(f"[VinDr-Kaggle] classi presenti: {found}; assenti (saltate): {absent}")
    print(f"[VinDr-Kaggle] classi del CSV non mappate (ignorate): {sorted(set(ann.loc[ann['target'].isna(), 'class_name']))}")
    thr = 2 if cfg["data"].get("kaggle_label_rule", "majority") == "majority" else 1
    nrad = ann[ann["target"].notna()].groupby(["image_id", "target"])["rad_id"].nunique()
    ids = sorted(geo)
    out = pd.DataFrame({"image_id": ids, "cache_path": [os.path.join(cdir, i + ".png") for i in ids]})
    g = np.array([geo[i] for i in ids])
    out["orig_w"], out["orig_h"], out["sx"], out["sy"], out["ox"], out["oy"] = g.T
    pos_sets = {}
    for t in found:
        pos = set(nrad.xs(t, level="target").loc[lambda s: s >= thr].index) if t in nrad.index.get_level_values(1) else set()
        pos_sets[t] = pos
        out[f"vd_{t}"] = out["image_id"].isin(pos).astype(int)
    out.to_csv(os.path.join(out_dir, "vindr_test.csv"), index=False)
    b = ann[ann["target"].notna() & ann["x_min"].notna()].copy()
    b = b[[iid in pos_sets[t] for iid, t in zip(b["image_id"], b["target"])]]
    G = np.array([geo[i] for i in b["image_id"]]) if len(b) else np.zeros((0, 6))
    bb = pd.DataFrame({"image_id": b["image_id"].values, "target": b["target"].values, "rad_id": b["rad_id"].values,
                       "x1": b["x_min"].values * G[:, 2] + G[:, 4], "y1": b["y_min"].values * G[:, 3] + G[:, 5],
                       "x2": b["x_max"].values * G[:, 2] + G[:, 4], "y2": b["y_max"].values * G[:, 3] + G[:, 5]})
    bb.to_csv(os.path.join(out_dir, "vindr_bbox_test.csv"), index=False)
    prev = {t: int(out[f"vd_{t}"].sum()) for t in found}
    print(f"[VinDr-Kaggle] {len(out)} immagini; positivi (regola >= {thr}/3): {prev}")
    print(f"[VinDr-Kaggle] box: {len(bb)} su {bb['image_id'].nunique()} immagini")
    return {"vindr_source": "kaggle", "vindr_label_rule": f">={thr}/3", "vindr_test_images": len(out),
            "vindr_positives": prev, "vindr_targets_absent": absent, "vindr_boxes": len(bb),
            "vindr_box_images": int(bb["image_id"].nunique())}


def prepare_vindr(cfg, out_dir, workers):
    if cfg["paths"].get("vindr_source", "physionet") == "kaggle":
        return prepare_vindr_kaggle(cfg, out_dir, workers)
    root = vindr_dir(cfg)
    dcm = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
           if p.lower().endswith((".dicom", ".dcm")) and os.sep + "test" + os.sep in p.lower()]
    print(f"[VinDr] DICOM test trovati: {len(dcm)}")
    if not dcm:
        print("[VinDr][ATTENZIONE] nessun DICOM: salto VinDr (la valutazione esterna non sara' disponibile).")
        return {"vindr_test_images": 0}
    cdir, geo = _cache_dicoms(dcm, out_dir, workers)

    lab = pd.read_csv(find_ci(root, "image_labels_test.csv"))
    cols = {norm_name(c): c for c in lab.columns}
    out = pd.DataFrame({"image_id": lab["image_id"].astype(str)})
    out = out[out["image_id"].isin(geo)].copy()
    lab = lab.set_index("image_id").loc[out["image_id"]]
    out["cache_path"] = [os.path.join(cdir, i + ".png") for i in out["image_id"]]
    g = np.array([geo[i] for i in out["image_id"]])
    out["orig_w"], out["orig_h"], out["sx"], out["sy"], out["ox"], out["oy"] = g.T
    missing = []
    for tgt, (vname, _) in VINDR_TARGETS.items():
        c = cols.get(norm_name(vname))
        if c is None:
            missing.append(vname)
            continue
        out[f"vd_{tgt}"] = (lab[c].values > 0).astype(int)
    if missing:
        raise RuntimeError(f"Colonne VinDr non trovate in image_labels_test.csv: {missing}. Colonne presenti: {list(lab.columns)}")
    out.to_csv(os.path.join(out_dir, "vindr_test.csv"), index=False)

    ann = pd.read_csv(find_ci(root, "annotations_test.csv"))
    name2tgt = {norm_name(v): t for t, (v, _) in VINDR_TARGETS.items() if t != "Pneumonia"}
    ann["target"] = ann["class_name"].map(lambda s: name2tgt.get(norm_name(s)))
    ann = ann[ann["target"].notna() & ann["x_min"].notna() & ann["image_id"].isin(geo)].copy()
    G = np.array([geo[i] for i in ann["image_id"]])
    sx, sy, ox, oy = G[:, 2], G[:, 3], G[:, 4], G[:, 5]
    bb = pd.DataFrame({"image_id": ann["image_id"].values, "target": ann["target"].values,
                       "x1": ann["x_min"].values * sx + ox, "y1": ann["y_min"].values * sy + oy,
                       "x2": ann["x_max"].values * sx + ox, "y2": ann["y_max"].values * sy + oy})
    bb.to_csv(os.path.join(out_dir, "vindr_bbox_test.csv"), index=False)
    prev = {t: int(out[f"vd_{t}"].sum()) for t in VINDR_TARGETS}
    print(f"[VinDr] {len(out)} immagini; positivi per target: {prev}")
    print(f"[VinDr] box mappati: {len(bb)} su {bb['image_id'].nunique()} immagini; per target: {bb['target'].value_counts().to_dict()}")
    return {"vindr_test_images": len(out), "vindr_positives": prev, "vindr_boxes": len(bb),
            "vindr_box_images": int(bb["image_id"].nunique()), "vindr_boxes_per_target": bb["target"].value_counts().to_dict()}


def main():
    global CACHE, GEOM
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--skip-nih", action="store_true")
    ap.add_argument("--skip-vindr", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    CACHE, GEOM = cfg["data"]["cache_size"], cfg["data"]["vindr_geometry"]
    out = shared_data_dir(cfg)
    rep = {}
    if not a.skip_nih:
        rep.update(prepare_nih(cfg, out, cfg["data"]["prep_workers"]))
    if not a.skip_vindr:
        rep.update(prepare_vindr(cfg, out, cfg["data"]["prep_workers"]))
    rep.update(cache_size=CACHE, vindr_geometry=GEOM)
    dump_json(rep, os.path.join(out, "prep_report.json"))
    print(f"[OK] manifest in {out}")


if __name__ == "__main__":
    main()
