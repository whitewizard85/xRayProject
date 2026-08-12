import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# =====================================================
# PATHS
# =====================================================
PYTHON_DIR = "/home/gpuvm/Desktop/Luca Migliaccio/PythonCode"
root_dir = "/home/gpuvm/Desktop/Luca Migliaccio/archive"

test_csv = os.path.join(PYTHON_DIR, "test_split.csv")
model_path = "best_densenet121_v2.pth"  # I pesi del tuo modello ottimale
threshold_path = os.path.join(PYTHON_DIR, "optimized_thresholds.json")

# =====================================================
# CLASSES (Le 14 classi del tuo modello attuale)
# =====================================================
classes = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
num_classes = len(classes)

# =====================================================
# LABEL ENCODING
# =====================================================
def encode_labels(label_str):
    vec = torch.zeros(num_classes)
    labels = label_str.split("|")
    for l in labels:
        if l in classes:
            vec[classes.index(l)] = 1.0
    return vec

# =====================================================
# IMAGE PATH
# =====================================================
def get_image_path(img_name):
    for i in range(1, 13):
        folder = f"images_{i:03d}"
        path = os.path.join(root_dir, folder, "images", img_name)
        if os.path.exists(path):
            return path
    return None

# =====================================================
# DATASET
# =====================================================
class NIHChestDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["Image Index"]
        label_str = row["Finding Labels"]

        img_path = get_image_path(img_name)
        if img_path is None:
            return None, None

        image = Image.open(img_path).convert("RGB")
        label = encode_labels(label_str)

        if self.transform:
            image = self.transform(image)

        return image, label

# =====================================================
# COLLATE FUNCTION
# =====================================================
def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0)
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return images, labels

# =====================================================
# IMAGE SIZE & TRANSFORMS (Allineato a 384)
# =====================================================
IMAGE_SIZE = 384

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# =====================================================
# DEVICE
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDEVICE:", device)

# =====================================================
# LOAD DATA
# =====================================================
test_df = pd.read_csv(test_csv)
test_dataset = NIHChestDataset(test_df, val_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn
)

print("Test batches:", len(test_loader))

# =====================================================
# MODEL
# =====================================================
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Non trovo i pesi del modello in: {model_path}")

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

print("Model loaded ✔")

# =====================================================
# THRESHOLDS
# =====================================================
with open(threshold_path, "r") as f:
    thresholds = json.load(f)

# =====================================================
# INFERENCE
# =====================================================
all_probs = []
all_targets = []
valid_indices = []  # Tiene traccia delle righe effettivamente caricate (escludendo i file mancanti)

print("\nRunning inference on Test Set...")

current_idx = 0
with torch.no_grad():
    for idx, (images, labels) in enumerate(tqdm(test_loader)):
        if images.numel() == 0:
            continue
            
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)

        all_probs.append(probs.cpu())
        all_targets.append(labels.cpu())
        
        # Registriamo l'indice reale del dataframe originale per mappare le immagini corrette
        for _ in range(images.size(0)):
            valid_indices.append(current_idx)
            current_idx += 1

all_probs = torch.cat(all_probs).numpy()
all_targets = torch.cat(all_targets).numpy()

# =====================================================
# APPLY THRESHOLDS
# =====================================================
preds = np.zeros_like(all_probs)
for j, c in enumerate(classes):
    t = thresholds.get(c, 0.5)
    preds[:, j] = (all_probs[:, j] >= t).astype(int)

# =====================================================
# ERROR ANALYSIS STORAGE
# =====================================================
fp_results = {c: [] for c in classes}
fn_results = {c: [] for c in classes}

# =====================================================
# ANALYSIS LOOP
# =====================================================
# Usiamo valid_indices per assicurarci che test_df.iloc[i] corrisponda all'immagine reale
for i in range(len(all_probs)):
    df_row_idx = valid_indices[i]
    img_id = test_df.iloc[df_row_idx]["Image Index"]

    for j, c in enumerate(classes):
        y_true = all_targets[i, j]
        y_pred = preds[i, j]
        prob = all_probs[i, j]

        # FALSE POSITIVE (Sano ma predetto Malato)
        if y_true == 0 and y_pred == 1:
            fp_results[c].append((img_id, float(prob)))

        # FALSE NEGATIVE (Malato ma predetto Sano)
        if y_true == 1 and y_pred == 0:
            fn_results[c].append((img_id, float(prob)))

# =====================================================
# REPORT PRINT & SORTING BY WORST CASES
# =====================================================
print("\n========================")
print("ERROR ANALYSIS REPORT (PEGIORI CASI ORDINATI)")
print("========================")

for c in classes:
    fps = fp_results[c]
    fns = fn_results[c]

    # Ordina i FP dal valore di probabilità più alto (errore più grave)
    fps_sorted = sorted(fps, key=lambda x: x[1], reverse=True)
    # Ordina i FN dal valore di probabilità più basso (cecità più totale)
    fns_sorted = sorted(fns, key=lambda x: x[1])

    print(f"\n{c}")
    print(f"  Total False Positives: {len(fps)}")
    print(f"  Total False Negatives: {len(fns)}")

    if len(fps_sorted) > 0:
        print("  Worst FP (Sicuro ma errato):", [(img, round(p, 4)) for img, p in fps_sorted[:3]])

    if len(fns_sorted) > 0:
        print("  Worst FN (Cieco ma malato): ", [(img, round(p, 4)) for img, p in fns_sorted[:3]])

# =====================================================
# SAVE CSV REPORT
# =====================================================
rows = []
for c in classes:
    for img, p in fp_results[c]:
        rows.append([img, c, "FP", p])

    for img, p in fn_results[c]:
        rows.append([img, c, "FN", p])

df_errors = pd.DataFrame(rows, columns=["image", "class", "error_type", "probability"])

save_path = os.path.join(PYTHON_DIR, "error_analysis.csv")
df_errors.to_csv(save_path, index=False)

print("\n========================")
print("REPORT DEGLI ERRORI SALVATO CON SUCCESSO ✔")
print("========================")
print(save_path)