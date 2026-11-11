"""
common.py -- utilita' condivise dalla pipeline v2.

Contiene: lettura config, classi e mappature NIH<->VinDr, dataset su cache,
trasformazioni, geometria (box -> spazio di input del modello), modelli e
target layer, loss, metriche di calibrazione e ricalibrazione post-hoc.
"""
import os
import json
import math
import random
import copy

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path=None, smoke=False):
    path = path or os.environ.get("XRAY_CONFIG", os.path.join(HERE, "config.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_smoke"] = bool(smoke)
    cfg["train"].setdefault("model_override", None)
    cfg["train"].setdefault("pretrained", True)
    if smoke:
        if cfg["smoke"].get("model_override"):
            cfg["train"]["model_override"] = cfg["smoke"]["model_override"]
        cfg["train"]["pretrained"] = cfg["smoke"].get("pretrained", True)
        cfg["paths"]["work_dir"] = os.path.join(cfg["paths"]["work_dir"], cfg["smoke"]["work_subdir"])
        cfg["experiment"]["seeds"] = cfg["smoke"]["seeds"]
        cfg["train"]["fixed_epochs"] = cfg["smoke"]["epochs"]
        cfg["train"]["max_epochs_cap"] = cfg["smoke"]["epochs"]
        a = cfg["attribution"]
        a["ig_max_pairs"] = min(a["ig_max_pairs"], 8)
        a["faithfulness_pairs"] = min(a["faithfulness_pairs"], 6)
        a["sanity_pairs"] = min(a["sanity_pairs"], 4)
        a["ig_steps"] = min(a["ig_steps"], 4)
        cfg["analysis"]["n_boot"] = 50
    os.makedirs(cfg["paths"]["work_dir"], exist_ok=True)
    return cfg


def wpath(cfg, *parts):
    p = os.path.join(cfg["paths"]["work_dir"], *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def shared_data_dir(cfg):
    """La cache dati e' condivisa fra run normali e smoke (la cache smoke e' un sottoinsieme)."""
    base = cfg["paths"]["work_dir"]
    if cfg["_smoke"]:
        base = os.path.dirname(base)
    d = os.path.join(base, "data")
    os.makedirs(d, exist_ok=True)
    return d


def vindr_dir(cfg):
    return os.path.join(cfg["paths"]["vindr_root"], "vindr-cxr", "1.0.0")


def dump_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------------
# Classi e mappature
# ---------------------------------------------------------------------
CLASSES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
           "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
           "Fibrosis", "Pleural_Thickening", "Hernia"]
NUM_CLASSES = len(CLASSES)
CIDX = {c: i for i, c in enumerate(CLASSES)}

# Nomi in BBox_List_2017.csv -> classi NIH
NIH_BBOX_TO_CLASS = {
    "Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Effusion": "Effusion",
    "Infiltrate": "Infiltration", "Infiltration": "Infiltration", "Mass": "Mass",
    "Nodule": "Nodule", "Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax",
}

# Target di valutazione esterna su VinDr-CXR: nome target -> (etichetta VinDr, classi NIH usate)
# Mass e Nodule NIH sono uniti perche' VinDr ha un'unica classe "Nodule/Mass".
# Hernia non ha corrispondente in VinDr ed e' esclusa.
VINDR_TARGETS = {
    "Atelectasis":        ("Atelectasis",        ["Atelectasis"]),
    "Cardiomegaly":       ("Cardiomegaly",       ["Cardiomegaly"]),
    "Effusion":           ("Pleural effusion",   ["Effusion"]),
    "Infiltration":       ("Infiltration",       ["Infiltration"]),
    "Nodule/Mass":        ("Nodule/Mass",        ["Mass", "Nodule"]),
    "Pneumothorax":       ("Pneumothorax",       ["Pneumothorax"]),
    "Consolidation":      ("Consolidation",      ["Consolidation"]),
    "Edema":              ("Edema",              ["Edema"]),
    "Emphysema":          ("Emphysema",          ["Emphysema"]),
    "Fibrosis":           ("Pulmonary fibrosis", ["Fibrosis"]),
    "Pleural_Thickening": ("Pleural thickening", ["Pleural_Thickening"]),
    "Pneumonia":          ("Pneumonia",          ["Pneumonia"]),   # etichetta globale: niente box
}


def norm_name(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def labels_from_string(s):
    v = np.zeros(NUM_CLASSES, dtype=np.float32)
    for l in str(s).split("|"):
        l = l.strip()
        if l in CIDX:
            v[CIDX[l]] = 1.0
    return v


# ---------------------------------------------------------------------
# Riproducibilita'
# ---------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Immagini -> uint8
# ---------------------------------------------------------------------
def to_uint8(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(-1) if arr.shape[-1] >= 3 else arr[..., 0]
    arr = arr.astype(np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def load_pil_gray_any(path):
    """Apre un PNG NIH gestendo modi L, RGB, RGBA, I;16 senza perdere dinamica."""
    img = Image.open(path)
    if img.mode == "L":
        return img.copy()
    if img.mode in ("RGB", "RGBA", "LA", "P"):
        return img.convert("L")
    return Image.fromarray(to_uint8(np.array(img)))


# ---------------------------------------------------------------------
# Dataset su cache
# ---------------------------------------------------------------------
class CachedDataset(torch.utils.data.Dataset):
    """df con colonne: image_id, cache_path, e colonne label (14 classi NIH) opzionali."""

    def __init__(self, df, transform, label_cols=None):
        self.paths = df["cache_path"].tolist()
        self.ids = df["image_id"].tolist()
        self.transform = transform
        if label_cols is not None:
            self.labels = df[label_cols].values.astype(np.float32)
        else:
            self.labels = np.zeros((len(df), NUM_CLASSES), dtype=np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), torch.from_numpy(self.labels[i]), i


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(tcfg, crop, aug):
    """aug: 0 = nessuna, 1 = flip orizzontale, 2 = augmentation "forte" (regime alternativo):
    flip + piccola trasformazione affine + variazioni di luminosita'/contrasto."""
    from torchvision import transforms as T
    S, R = tcfg["image_size"], tcfg["crop_resize"]
    base = [T.Resize((R, R)), T.CenterCrop((S, S))] if crop else [T.Resize((S, S))]
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if aug == 2:
        extra = [T.RandomHorizontalFlip(p=0.5),
                 T.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.93, 1.07)),
                 T.ColorJitter(brightness=0.15, contrast=0.15)]
    elif aug:
        extra = [T.RandomHorizontalFlip(p=0.5)]
    else:
        extra = []
    train = list(base) + extra + norm
    return T.Compose(train), T.Compose(list(base) + norm)


def make_loader(ds, batch_size, shuffle, num_workers, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker, generator=g,
        persistent_workers=False, drop_last=False)


# ---------------------------------------------------------------------
# Geometria: box in coordinate cache -> spazio di input del modello
# ---------------------------------------------------------------------
def box_cache_to_input(box, cache_size, tcfg, crop):
    """box = (x1, y1, x2, y2) in pixel della cache (cache_size x cache_size).
    Ritorna box in pixel di input (image_size x image_size), clippato, oppure None
    se il box cade interamente fuori dalla regione vista dal modello."""
    S, R = tcfg["image_size"], tcfg["crop_resize"]
    x1, y1, x2, y2 = box
    if crop:
        s = R / cache_size
        off = (R - S) / 2.0
        x1, y1, x2, y2 = x1 * s - off, y1 * s - off, x2 * s - off, y2 * s - off
    else:
        s = S / cache_size
        x1, y1, x2, y2 = x1 * s, y1 * s, x2 * s, y2 * s
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(S), x2), min(float(S), y2)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return (x1, y1, x2, y2)


def boxes_to_mask(boxes, S):
    m = np.zeros((S, S), dtype=bool)
    for (x1, y1, x2, y2) in boxes:
        m[int(math.floor(y1)):int(math.ceil(y2)), int(math.floor(x1)):int(math.ceil(x2))] = True
    return m


def common_region_bounds(tcfg):
    """Regione dell'immagine vista da TUTTE le condizioni (quella del center crop),
    in frazione del lato dell'immagine completa."""
    S, R = tcfg["image_size"], tcfg["crop_resize"]
    off = (R - S) / 2.0
    return off / R, (off + S) / R


def to_common_grid(maps, tcfg, crop, grid):
    """maps: tensor (B, S, S) in spazio di input. Ritorna (B, grid, grid) sulla regione comune."""
    S = tcfg["image_size"]
    x = maps.unsqueeze(1)
    if not crop:
        lo, hi = common_region_bounds(tcfg)
        a, b = int(round(lo * S)), int(round(hi * S))
        x = x[:, :, a:b, a:b]
    x = F.interpolate(x, size=(grid, grid), mode="area")
    return x[:, 0]


# ---------------------------------------------------------------------
# Modelli
# ---------------------------------------------------------------------
TIMM_NAMES = {"convnext_base": "convnext_base.fb_in22k", "densenet121": "densenet121.tv_in1k",
              # estensione: transformer gerarchico pre-addestrato su ImageNet-22k a 384 px (come ConvNeXt-Base IN-22k)
              "swin_base": "swin_base_patch4_window12_384.ms_in22k"}


def build_model(arch, pretrained=True, override=None):
    import timm
    name = override or TIMM_NAMES[arch]
    return timm.create_model(name, pretrained=pretrained, num_classes=NUM_CLASSES)


def is_swin(model):
    return hasattr(model, "layers") and hasattr(model, "patch_embed")


def feature_layout(model):
    """Layout dell'output del target layer: Swin (timm) produce mappe (B, H, W, C)."""
    return "nhwc" if is_swin(model) else "nchw"


def target_layer(model):
    """Ultimo stadio (mappa spaziale) per Grad-CAM. Per Swin l'output e' NHWC (vedi feature_layout)."""
    if is_swin(model):                      # Swin Transformer (timm): ultimo stadio gerarchico
        return model.layers[-1]
    if hasattr(model, "stages"):            # ConvNeXt (timm)
        return model.stages[-1]
    if hasattr(model, "features"):          # DenseNet (timm): features termina con norm5 (BN+ReLU)
        return model.features
    raise ValueError("target layer non definito per questo modello")


def last_block_and_head(model):
    """Moduli da re-inizializzare nel test di randomizzazione (cascata dall'alto, primo livello)."""
    mods = []
    if is_swin(model):
        mods.append(model.layers[-1])
    elif hasattr(model, "stages"):
        mods.append(model.stages[-1])
    elif hasattr(model, "features"):
        blk = getattr(model.features, "denseblock4", None)
        mods.append(blk if blk is not None else model.features)
    head = model.get_classifier()
    mods.append(head)
    return mods


def cascade_levels(model):
    """Livelli per la randomizzazione A CASCATA (Adebayo et al., 2018), dall'alto verso il basso.
    Ogni elemento: (nome, [moduli]). La randomizzazione e' cumulativa: al livello k sono
    re-inizializzati tutti i livelli 0..k."""
    head = model.get_classifier()
    if is_swin(model):
        lv = [("head", [head, getattr(model, "norm", head)])]
        n = len(model.layers)
        lv += [(f"layers.{i}", [model.layers[i]]) for i in range(n - 1, -1, -1)]
        lv += [("patch_embed", [model.patch_embed])]
        return lv
    if hasattr(model, "stages"):            # ConvNeXt
        lv = [("head", [model.head])]
        n = len(model.stages)
        lv += [(f"stages.{i}", [model.stages[i]]) for i in range(n - 1, -1, -1)]
        lv += [("stem", [model.stem])]
        return lv
    if hasattr(model, "features"):          # DenseNet (timm/torchvision naming)
        f = model.features
        lv = [("classifier", [head] + ([f.norm5] if hasattr(f, "norm5") else []))]
        for k in [4, 3, 2, 1]:
            lv.append((f"denseblock{k}", [getattr(f, f"denseblock{k}")]))
            if k > 1 and hasattr(f, f"transition{k - 1}"):
                lv.append((f"transition{k - 1}", [getattr(f, f"transition{k - 1}")]))
        stem = [getattr(f, n) for n in ("conv0", "norm0") if hasattr(f, n)]
        if stem:
            lv.append(("stem", stem))
        return lv
    raise ValueError("livelli di cascata non definiti per questo modello")


@torch.no_grad()
def reinit_modules(modules, seed=0):
    """Re-inizializzazione uniforme per tutte le architetture (randomizzazione a cascata):
      - pesi convoluzionali (4D): He/Kaiming normale, std = sqrt(2 / fan_in) (scala del segnale preservata);
      - altri pesi >= 2D (lineari, tabelle di bias posizionale): N(0, 0.02) troncata a +-2 sigma;
      - pesi di normalizzazione = 1, bias = 0, altri parametri 1D (es. layer scale) ~ N(0, 0.02);
      - BatchNorm: statistiche di popolazione azzerate (media 0, varianza 1), come in un modello appena creato.
    Senza l'ultimo punto, in eval i BatchNorm userebbero le statistiche del modello addestrato su attivazioni
    ormai diverse e il segnale puo' divergere (mappe non finite), come osservato su DenseNet."""
    g = torch.Generator().manual_seed(seed)
    for m in modules:
        for sub in m.modules():
            if isinstance(sub, nn.modules.batchnorm._BatchNorm) and sub.track_running_stats:
                sub.reset_running_stats()
        for name, p in m.named_parameters():
            leaf = name.split(".")[-1]
            if p.ndim == 4:
                fan_in = p.shape[1] * p.shape[2] * p.shape[3]
                v = torch.randn(p.shape, generator=g) * math.sqrt(2.0 / max(fan_in, 1))
                p.copy_(v.to(p.dtype))
            elif p.ndim >= 2:
                v = torch.randn(p.shape, generator=g).clamp_(-2, 2) * 0.02
                p.copy_(v.to(p.dtype))
            elif leaf == "bias":
                p.zero_()
            elif leaf == "weight":
                p.fill_(1.0)
            else:
                p.copy_((torch.randn(p.shape, generator=g) * 0.02).to(p.dtype))


def randomize_modules(model, modules, seed=0):
    torch.manual_seed(seed)
    for m in modules:
        for sub in m.modules():
            if hasattr(sub, "reset_parameters"):
                try:
                    sub.reset_parameters()
                except Exception:
                    pass
    return model


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    """Identica allo script del paper: gamma_neg=4, gamma_pos=1, clip=0.05."""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, xs, ys):
        p = torch.sigmoid(xs)
        p_neg = 1.0 - p
        if self.clip and self.clip > 0:
            p_neg = (p_neg + self.clip).clamp(max=1.0)
        lp = ys * torch.log(p.clamp(min=self.eps)) * ((1 - p) ** self.gamma_pos)
        ln = (1 - ys) * torch.log(p_neg.clamp(min=self.eps)) * ((1 - p_neg) ** self.gamma_neg)
        return -(lp + ln).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, xs, ys):
        bce = F.binary_cross_entropy_with_logits(xs, ys, reduction="none")
        p = torch.sigmoid(xs)
        pt = ys * p + (1 - ys) * (1 - p)
        return ((1 - pt) ** self.gamma * bce).mean()


def build_loss(name, pos_weight=None):
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "wbce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "asl":
        return AsymmetricLoss()
    if name == "focal":
        return FocalLoss(2.0)
    raise ValueError(name)


# ---------------------------------------------------------------------
# Metriche di calibrazione
# ---------------------------------------------------------------------
def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def ece_uniform(p, y, n_bins=10):
    """ECE standard: sum_b (n_b/N) |acc_b - conf_b|, bin uniformi su [0,1]."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def ece_adaptive(p, y, n_bins=15):
    """ECE a massa uguale (bin con lo stesso numero di campioni)."""
    order = np.argsort(p)
    chunks = np.array_split(order, n_bins)
    return float(sum(len(c) / len(p) * abs(y[c].mean() - p[c].mean()) for c in chunks if len(c)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def nll(p, y, eps=1e-7):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calib_slope_intercept(p, y, eps=1e-7):
    """Regressione logistica y ~ a + b*logit(p). b=1, a=0 => calibrazione perfetta."""
    from sklearn.linear_model import LogisticRegression
    if y.min() == y.max():
        return np.nan, np.nan
    z = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1))
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def calib_metrics(p, y):
    s, a = calib_slope_intercept(p, y)
    return dict(ece=ece_uniform(p, y), ece_adaptive=ece_adaptive(p, y), brier=brier(p, y),
                nll=nll(p, y), slope=s, intercept=a)


def fit_temperature(z, y):
    """Temperature scaling per una classe: p = sigmoid(z / T). Ritorna T."""
    zt = torch.tensor(z, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    logt = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(zt / torch.exp(logt), yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.exp(logt).item())


def fit_platt(z, y):
    """Platt scaling per una classe: p = sigmoid(a*z + b). Ritorna (a, b)."""
    zt = torch.tensor(z, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    ab = torch.tensor([1.0, 0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(ab[0] * zt + ab[1], yt)
        loss.backward()
        return loss
    opt.step(closure)
    return float(ab[0].item()), float(ab[1].item())


def youden_threshold(p, y):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


# ---------------------------------------------------------------------
# Manifest dati (creati da 02_prepare_data.py)
# ---------------------------------------------------------------------
def load_manifest(cfg, name):
    p = os.path.join(shared_data_dir(cfg), f"{name}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} non trovato: esegui prima 02_prepare_data.py")
    df = pd.read_csv(p)
    if cfg["_smoke"]:
        n = {"nih_train": cfg["smoke"]["n_train"], "nih_val": cfg["smoke"]["n_val"],
             "nih_test": cfg["smoke"]["n_test"], "vindr_test": cfg["smoke"]["n_vindr"]}.get(name)
        if n is not None and len(df) > n:
            df = df.sample(n=n, random_state=0).reset_index(drop=True)
    return df


def run_dir(cfg, run_id):
    d = os.path.join(cfg["paths"]["work_dir"], "runs", run_id)
    os.makedirs(d, exist_ok=True)
    return d


def autocast_ctx(cfg, device):
    amp = cfg["train"]["amp"]
    if device.type != "cuda" or amp == "none":
        return torch.autocast(device_type="cpu", enabled=False)
    dt = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dt)


def load_model_for_run(cfg, run, device):
    model = build_model(run["arch"], pretrained=False, override=cfg["train"]["model_override"])
    sd = torch.load(os.path.join(run_dir(cfg, run["run_id"]), "best.pt"), map_location="cpu")
    model.load_state_dict(sd)
    return model.to(device).eval()


@torch.no_grad()
def predict_logits(model, loader, device, cfg):
    model.eval()
    out = []
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        if cfg["train"]["channels_last"] and device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        with autocast_ctx(cfg, device):
            z = model(x)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, 0)
