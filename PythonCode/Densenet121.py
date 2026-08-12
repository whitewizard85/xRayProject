import torch
import torch.nn as nn
from torchvision import models


# =========================
# 1. DEVICE SETUP
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\n========================")
print("DEVICE")
print("========================")
print(device)


# =========================
# 2. NIH CLASSES
# =========================

classes = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

num_classes = len(classes)


# =========================
# 3. LOAD PRETRAINED DENSENET121
# =========================

print("\n========================")
print("LOADING DENSENET121")
print("========================")

model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

print("Pretrained ImageNet weights loaded ✔")


# =========================
# 4. MODIFY FINAL CLASSIFIER
# =========================

in_features = model.classifier.in_features

print("\nOriginal classifier input features:", in_features)

model.classifier = nn.Linear(in_features, num_classes)

print("Final layer replaced ✔")
print("Output classes:", num_classes)


# =========================
# 5. MOVE MODEL TO GPU
# =========================

model = model.to(device)

print("\nModel moved to device ✔")


# =========================
# 6. LOSS FUNCTION
# =========================

criterion = nn.BCEWithLogitsLoss()

print("\nLoss function:")
print("BCEWithLogitsLoss ✔")


# =========================
# 7. FORWARD PASS TEST
# =========================

print("\n========================")
print("FORWARD PASS TEST")
print("========================")

# Fake batch
x = torch.randn(8, 3, 224, 224).to(device)

# Forward
with torch.no_grad():
    outputs = model(x)

print("Input shape:", x.shape)
print("Output shape:", outputs.shape)

# Should be [8, 14]
print("\nForward pass successful ✔")


# =========================
# 8. GPU MEMORY CHECK
# =========================

if torch.cuda.is_available():

    print("\n========================")
    print("GPU INFO")
    print("========================")

    print("GPU:", torch.cuda.get_device_name(0))

    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2

    print(f"Allocated memory: {allocated:.2f} MB")
    print(f"Reserved memory: {reserved:.2f} MB")


# =========================
# 9. MODEL SUMMARY
# =========================

print("\n========================")
print("MODEL READY")
print("========================")

print(model)