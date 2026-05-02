"""
MoleScan ViT-B/16 fine-tuning on ISIC 2019.

This script is designed to run end-to-end inside a Kaggle notebook with a
T4 or P100 GPU. The output is a single .pt file containing the fine-tuned
state_dict, which the FastAPI backend loads at startup.

Class mapping (ISIC 2019 8-class → MoleScan 3-class):
    healthy:    NV, BKL, DF, VASC
    suspicious: AK, BCC
    malignant:  MEL, SCC

Hyperparameters are standard ViT fine-tuning defaults — see the README for
justification. Everything is configurable through the Config class.

Output (saved to /kaggle/working/):
    molescan_vit.pt        - fine-tuned state_dict (drop into backend/weights/)
    training_history.json  - per-epoch metrics
    final_metrics.json     - test set metrics including ECE
    confusion_matrix.png   - test set confusion matrix
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    get_cosine_schedule_with_warmup,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Paths — VERIFY THESE ON KAGGLE before running.
    # The exact path depends on which ISIC 2019 dataset you attach.
    # Common patterns:
    #   /kaggle/input/isic-2019/ISIC_2019_Training_Input/...
    #   /kaggle/input/isic-2019/ISIC_2019_Training_GroundTruth.csv
    image_dir: str = "/kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
    labels_csv: str = "/kaggle/input/isic-2019/ISIC_2019_Training_GroundTruth.csv"
    out_dir: str = "/kaggle/working"

    # Model
    model_name: str = "google/vit-base-patch16-224"
    num_classes: int = 3
    image_size: int = 224

    # Training
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 10
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0

    # Split
    val_size: float = 0.10
    test_size: float = 0.10
    seed: int = 42


CFG = Config()


# Class mapping — ISIC 2019 → MoleScan
ISIC_8_TO_3 = {
    "MEL": "malignant",
    "SCC": "malignant",
    "BCC": "suspicious",
    "AK": "suspicious",
    "NV": "healthy",
    "BKL": "healthy",
    "DF": "healthy",
    "VASC": "healthy",
}
CLASS_NAMES = ["healthy", "suspicious", "malignant"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_labels(csv_path: str) -> pd.DataFrame:
    """Load ISIC 2019 ground truth CSV and map 8 classes → 3.

    The CSV has columns: image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK
    Each row has exactly one column set to 1.0 (one-hot).
    """
    df = pd.read_csv(csv_path)
    label_cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

    # Drop rows labelled UNK or with no positive label
    df = df[df[label_cols].sum(axis=1) == 1].copy()

    df["isic_label"] = df[label_cols].idxmax(axis=1)
    df["molescan_label"] = df["isic_label"].map(ISIC_8_TO_3)
    df["label_idx"] = df["molescan_label"].map(CLASS_TO_IDX)

    print(f"Loaded {len(df)} labelled images.")
    print("Class distribution (3-class):")
    print(df["molescan_label"].value_counts())
    print("Original ISIC distribution:")
    print(df["isic_label"].value_counts())

    return df[["image", "isic_label", "molescan_label", "label_idx"]].reset_index(drop=True)


def stratified_split(df: pd.DataFrame, cfg: Config):
    """80/10/10 train/val/test, stratified by 3-class label."""
    train_val_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        stratify=df["label_idx"],
        random_state=cfg.seed,
    )
    val_relative = cfg.val_size / (1.0 - cfg.test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative,
        stratify=train_val_df["label_idx"],
        random_state=cfg.seed,
    )
    print(f"Split sizes — train: {len(train_df)}  val: {len(val_df)}  test: {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class ISIC2019Dataset(Dataset):
    """ISIC 2019 dataset wrapping the official ground truth CSV."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        processor: ViTImageProcessor,
        augment: bool = False,
        image_size: int = 224,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.image_size = image_size

        if augment:
            # Conservative medical-imaging augmentations.
            # Moles are rotation-invariant so flips/rotations are safe.
            # Mild color jitter compensates for lighting differences.
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ])
        else:
            self.transform = transforms.Resize((image_size, image_size))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.image_dir / f"{row['image']}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # ViTImageProcessor handles ImageNet normalization to match the
        # checkpoint distribution. We pass the already-resized PIL image.
        pixel_values = self.processor(
            images=img, return_tensors="pt", do_resize=False
        )["pixel_values"].squeeze(0)

        return pixel_values, int(row["label_idx"])


def compute_class_weights(df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for weighted cross-entropy."""
    counts = df["label_idx"].value_counts().sort_index().to_numpy()
    weights = counts.sum() / (num_classes * counts)
    print(f"Class weights: {dict(zip(CLASS_NAMES, weights.round(3)))}")
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (Guo et al., ICML 2017).

    Same definition the thesis uses (§5.1). Bins predictions by max-softmax
    confidence and compares average confidence to average accuracy in each bin.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += (mask.sum() / n) * abs(avg_conf - avg_acc)

    return float(ece)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Run full evaluation: accuracy, per-class F1, macro F1, ECE, confusion matrix."""
    model.eval()
    all_probs, all_labels = [], []

    for pixel_values, labels in tqdm(loader, desc="eval", leave=False):
        pixel_values = pixel_values.to(device)
        logits = model(pixel_values=pixel_values).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "per_class_f1": f1_score(labels, preds, average=None).tolist(),
        "ece": compute_ece(probs, labels),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(
            labels, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    running_loss = 0.0
    n_samples = 0

    for pixel_values, labels in tqdm(loader, desc="train", leave=False):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        logits = model(pixel_values=pixel_values).logits
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * pixel_values.size(0)
        n_samples += pixel_values.size(0)

    return running_loss / n_samples


def main(cfg: Config = CFG) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    df = load_labels(cfg.labels_csv)
    train_df, val_df, test_df = stratified_split(df, cfg)

    processor = ViTImageProcessor.from_pretrained(cfg.model_name)
    train_ds = ISIC2019Dataset(train_df, cfg.image_dir, processor, augment=True, image_size=cfg.image_size)
    val_ds = ISIC2019Dataset(val_df, cfg.image_dir, processor, augment=False, image_size=cfg.image_size)
    test_ds = ISIC2019Dataset(test_df, cfg.image_dir, processor, augment=False, image_size=cfg.image_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = ViTForImageClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Optimizer / scheduler / loss ─────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    class_weights = compute_class_weights(train_df, cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Train loop ───────────────────────────────────────────────────────────
    history = []
    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, cfg.grad_clip
        )
        val_metrics = evaluate(model, val_loader, device)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_ece": val_metrics["ece"],
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_log)
        print(
            f"[epoch {epoch:02d}] "
            f"loss={train_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"val_f1={val_metrics['macro_f1']:.4f}  "
            f"val_ece={val_metrics['ece']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  ✓ new best (macro_f1={best_val_f1:.4f})")

    # ── Test on best checkpoint ──────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print("\n=== TEST METRICS ===")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Per-class F1: {dict(zip(CLASS_NAMES, [round(x, 4) for x in test_metrics['per_class_f1']]))}")
    print(f"ECE: {test_metrics['ece']:.4f}")
    print(f"Confusion matrix:\n{np.array(test_metrics['confusion_matrix'])}")

    # ── Save artefacts ───────────────────────────────────────────────────────
    weights_path = out_dir / "molescan_vit.pt"
    torch.save(best_state, weights_path)
    print(f"\nWeights saved to {weights_path}")

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        cm = np.array(test_metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Test confusion matrix")
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
        plt.close()
        print(f"Confusion matrix saved to {out_dir / 'confusion_matrix.png'}")
    except Exception as exc:
        print(f"(Skipped confusion matrix plot: {exc})")

    print("\nDone. Download molescan_vit.pt from /kaggle/working/ and drop it "
          "into the backend's weights/ directory.")


if __name__ == "__main__":
    main()