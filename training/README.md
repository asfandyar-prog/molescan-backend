# MoleScan Training — Kaggle

End-to-end ViT-B/16 fine-tuning on ISIC 2019, mapped to MoleScan's 3-class
schema (healthy / suspicious / malignant). Output is a single `.pt` file
that drops into the FastAPI backend's `weights/` directory.

## Why Kaggle

- Free Tesla T4 / P100 GPU (30 hours per week)
- ISIC 2019 is hosted on Kaggle — no 30 GB download to your PC
- Saves trained weights to `/kaggle/working/` — you only download a ~340 MB `.pt` file
- Reproducible, shareable notebook environment

## Setup

1. **Sign in** to https://kaggle.com (free account).
2. **Verify your phone number** under Account Settings — required for GPU access.
3. **Create a new notebook**: Code → New Notebook.
4. **Add the dataset**: in the notebook sidebar, click "Add Data" → search "ISIC 2019" → add the official one (≈9 GB, 25,331 images).
5. **Enable GPU**: Settings (right sidebar) → Accelerator → GPU T4 x2 *or* GPU P100.
6. **Set Internet to On** under Settings (needed to download the ViT checkpoint from HuggingFace on first run).

## Running

Two options:

### Option A — paste the script as one cell

1. Copy the contents of `train_isic.py`.
2. Paste into a single Kaggle notebook cell.
3. **Verify the dataset paths** at the top of the script. Common patterns:

   ```
   /kaggle/input/isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input/
   /kaggle/input/isic-2019/ISIC_2019_Training_GroundTruth.csv
   ```

   Run a quick `!ls /kaggle/input/isic-2019/` cell first to check the
   exact directory structure, then update `Config.image_dir` and
   `Config.labels_csv` if needed.
4. Run the cell. Expect ~30–60 min per epoch on a T4, so ~5–10 hours total
   for 10 epochs. Stay within Kaggle's 9-hour session limit; reduce
   `Config.epochs` if needed.

### Option B — upload the script as a file

1. On Kaggle: notebook → File → Upload → select `train_isic.py`.
2. In a cell run: `%run train_isic.py`

## Outputs

When training finishes, `/kaggle/working/` will contain:

| File | What it is |
|------|------------|
| `molescan_vit.pt` | Fine-tuned weights — drop into `weights/` in the backend |
| `training_history.json` | Per-epoch loss / accuracy / F1 / ECE |
| `final_metrics.json` | Test set accuracy, macro F1, per-class F1, ECE, confusion matrix |
| `confusion_matrix.png` | Visualization of the test confusion matrix |

Right-click each file in the Kaggle file browser → Download.

## Hyperparameter notes

The defaults in `Config` are standard ViT fine-tuning values; nothing
exotic. Reasonable starting point — adjust based on actual training behaviour:

- `lr=5e-5` is conservative; if val loss plateaus very early, try `1e-4`.
- `epochs=10` is usually enough; watch the per-epoch logs for plateau or
  overfitting (val F1 drops while train loss keeps falling).
- `batch_size=32` fits the T4. If you switch to a smaller GPU, drop to 16.
- Class weights are computed automatically from the training split.

## Important: nothing about TTA happens here

This script produces the **base fine-tuned model**. The TTA wrapper
(`app/models/tta.py`) is applied at inference time inside the FastAPI
backend. The `.pt` file produced here is just the underlying model that
the wrapper loads.

## After training

```bash
# Locally:
# Drop the downloaded molescan_vit.pt into weights/ and restart the server.
mv ~/Downloads/molescan_vit.pt E:\molescan-backend\weights\
```

The backend's startup log will confirm:

```
Fine-tuned weights loaded from weights\molescan_vit.pt
```

Then `/predict` returns clinically-meaningful classifications instead of
random ImageNet-head outputs.