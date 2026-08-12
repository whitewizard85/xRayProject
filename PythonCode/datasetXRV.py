import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchxrayvision as xrv

# =========================
# CLASSI NIH (15)
# =========================
classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia","No Finding"
]

num_classes = len(classes)

# =========================
# LABEL ENCODING
# =========================
def encode_labels(label_str):
    vec = torch.zeros(num_classes, dtype=torch.float32)

    labels = label_str.split("|")

    # gestione "No Finding"
    if "No Finding" in labels:
        vec[-1] = 1.0
        return vec

    for l in labels:
        if l in classes:
            vec[classes.index(l)] = 1.0

    return vec

# =========================
# IMAGE FINDER NIH
# =========================
def get_image_path(img_name, image_root):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(image_root, folder, "images", img_name)

        if os.path.exists(path):
            return path

    return None

# =========================
# DATASET TORCHXRAYVISION
# =========================
class NIHChestXrayDataset(Dataset):

    def __init__(self, dataframe, image_root):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = image_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_name = row["Image Index"]
        label_str = row["Finding Labels"]

        img_path = get_image_path(img_name, self.image_root)

        if img_path is None:
            return None, None

        # =========================
        # LOAD GRAYSCALE (IMPORTANT)
        # =========================
        img = Image.open(img_path).convert("L")
        img = np.array(img).astype(np.float32)

        # =========================
        # NORMALIZZAZIONE XRV
        # =========================
        img = xrv.datasets.normalize(img, 255)

        # =========================
        # TO TENSOR (1, H, W)
        # =========================
        img = torch.from_numpy(img)[None, :, :]

        # =========================
        # LABEL
        # =========================
        label = encode_labels(label_str)

        return img, label

# =========================
# COLLATE FUNCTION SAFE
# =========================
def collate_fn(batch):

    batch = [b for b in batch if b[0] is not None]

    if len(batch) == 0:
        return None, None

    images = torch.stack([b[0] for b in batch])  # (B,1,H,W)
    labels = torch.stack([b[1] for b in batch])  # (B,C)

    return images, labels