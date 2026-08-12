import os
import torch
from torch.utils.data import Dataset
from PIL import Image

# =========================
# LABELS (14 CLASSI — NO FINDING RIMOSSA)
# =========================
classes = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration",
    "Mass","Nodule","Pneumonia","Pneumothorax",
    "Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia"
]

num_classes = len(classes)

# =========================
# ENCODING (MULTI-LABEL CLEAN)
# =========================
def encode_labels(label_str):

    vec = torch.zeros(num_classes, dtype=torch.float32)
    labels = label_str.split("|")

    # 🔴 IGNORA COMPLETAMENTE "No Finding"
    # (non viene più usato come classe)

    for l in labels:
        if l in classes:
            vec[classes.index(l)] = 1.0

    return vec

# =========================
# IMAGE PATH
# =========================
def get_image_path(img_name, image_root):

    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(image_root, folder, "images", img_name)

        if os.path.exists(path):
            return path

    return None

# =========================
# DATASET CLASS (CLEAN VERSION)
# =========================
class NIHChestXrayDataset(Dataset):

    def __init__(self, dataframe, image_root, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_name = row["Image Index"]
        label_str = row["Finding Labels"]

        img_path = get_image_path(img_name, self.image_root)

        if img_path is None:
            return None, None

        image = Image.open(img_path).convert("RGB")
        label = encode_labels(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label

# =========================
# COLLATE FUNCTION
# =========================
def collate_fn(batch):

    batch = [b for b in batch if b[0] is not None]

    if len(batch) == 0:
        return None, None

    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])

    return images, labels