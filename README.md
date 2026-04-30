# 🔬 MoleScan Backend

Dermatology mole classification API for the MoleScan mobile app.
Built with FastAPI + ViT, supervised by **Prof. Balázs Harangi** (University of Debrecen).
Target venue: **CITDS 2026** — submission deadline **31 May 2026**.

---

## Architecture

```
Mobile App (Saw)
    │  JPEG/PNG + metadata
    ▼
FastAPI  /predict
    │
ViT-B/16 fine-tuned on ISIC dataset
    │  LayerNorm TTA  ← novel contribution
    ▼
{ prediction, confidence, uncertainty, recommendation }
```

**Classes:** `healthy` · `suspicious` · `malignant`

---

## Quickstart (Windows / PowerShell)

### 1. Clone & enter the project

```powershell
git clone https://github.com/asfandyar-prog/molescan-backend.git
cd molescan-backend
```

### 2. Create virtual environment with uv

```powershell
# Install uv if needed
pip install uv

uv venv --python 3.11
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
uv pip install -e ".[dev]"
```

### 4. Configure environment

```powershell
Copy-Item .env.example .env
# Edit .env as needed
```

### 5. Run the server

```powershell
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness / readiness probe |
| `POST` | `/predict` | Classify a mole image |

### POST `/predict`

**Form fields:**
- `file` — JPEG or PNG image
- `location` — body location string (e.g. `"left forearm"`)
- `picture_date` — capture date (`YYYY-MM-DD`)

**Response:**
```json
{
  "prediction": "suspicious",
  "confidence": 0.7821,
  "uncertainty": "medium",
  "recommendation": "This mole shows features that warrant professional evaluation...",
  "location": "left forearm",
  "picture_date": "2026-04-30"
}
```

---

## Docker

```powershell
docker build -t molescan-backend .
docker run -p 8000:8000 -v ${PWD}/weights:/app/weights molescan-backend
```

---

## Project Structure

```
molescan-backend/
├── app/
│   ├── main.py                  # FastAPI app factory + lifespan
│   ├── core/
│   │   └── config.py            # Pydantic settings
│   ├── api/routes/
│   │   ├── health.py            # GET /health
│   │   └── predict.py           # POST /predict
│   ├── models/
│   │   └── classifier.py        # ViT + LayerNorm TTA
│   └── schemas/
│       └── prediction.py        # Request / response models
├── weights/                     # Fine-tuned .pt file (git-ignored)
├── pyproject.toml
├── Dockerfile
└── .env.example
```

---

## Novel Contribution — LayerNorm TTA

At inference time, **only the LayerNorm affine parameters** (γ, β) are
updated for a few gradient steps using entropy minimisation as a surrogate
loss. This adapts the backbone to the distribution shift introduced by the
digital microscope hardware without retraining the full model.

See `app/models/classifier.py → _layer_norm_tta()` for the implementation
stub and references.

---

## Developer

**Asfand Yar** · [@asfandyar-prog](https://github.com/asfandyar-prog)
