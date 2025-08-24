![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![HuggingFace](https://img.shields.io/badge/Transformers-CLIP-orange)

# Multimodal Social Activity Recommendation Engine 🏋️🎶📸

Recommend real-world activities (sports, yoga, events) from **images + text** using **CLIP** embeddings and **FAISS** similarity search.  
FastAPI microservice exposes `/recommend/text` and `/recommend/image`.

---

## ⚙️ Stack
- **Embeddings:** `sentence-transformers` (`clip-ViT-B-32`) for text & images  
- **Vector search:** `faiss-cpu`  
- **API:** FastAPI + Uvicorn  
- **Deployment:** Docker-ready  

---

## 🚀 Quickstart

```bash
# 1) Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Add datasets (Stanford-40, Yoga-82, etc.)
#    Expected format: data/raw/<label_name>/*.jpg
#    Example: data/raw/yoga_pose/img1.jpg

# 4) Build index
python scripts/build_index.py

# 5) Run API
uvicorn app.main:api --reload --port 8000
````

---

## 🧪 Example Usage

### Text recommendation

```bash
curl -X POST "http://localhost:8000/recommend/text" \
  -H "Content-Type: application/json" \
  -d '{"query": "beginner yoga class for weekends in Bangalore", "k": 5}'
```

### Image recommendation

```bash
curl -X POST "http://localhost:8000/recommend/image?k=5" \
  -F "file=@/path/to/yoga_pose.jpg"
```

---

## 🐳 Docker

```bash
docker build -t multimodal-recsys .
docker run -p 8000:8000 multimodal-recsys
```

---

## 📂 Data Layout

```
data/
  raw/
    yoga/
      img001.jpg
      ...
    playing_guitar/
      ...
  processed/
    embeddings.faiss
    items.json
```

> Extend `items.json` with metadata (city, price, schedule, etc.) to enable filtered recommendations.

---

## 🔑 TL;DR

1. Put labeled images in `data/raw/<label>/...jpg`
2. Run: `python scripts/build_index.py` → creates `data/processed/embeddings.faiss` + `items.json`
3. Start API: `uvicorn app.main:api --port 8000`
4. Query endpoints:

   * `POST /recommend/text` → `{"query":"suggest weekend dance classes in Bangalore","k":5}`
   * `POST /recommend/image` → file upload

```
```


---

# 📦 Repo layout

```
multimodal-recsys/
├─ README.md
├─ requirements.txt
├─ Dockerfile
├─ .gitignore
├─ data/
│  ├─ raw/                # put Stanford-40 / Yoga-82 images here
│  └─ processed/          # auto-created: embeddings + metadata
├─ app/
│  ├─ main.py             # FastAPI service
│  ├─ recommender.py      # CLIP + FAISS wrapper
│  ├─ ingest.py           # builds index from images + labels
│  ├─ schemas.py          # pydantic request/response models
│  └─ utils.py            # small helpers (image loading, paths, etc.)
└─ scripts/
   └─ build_index.py      # CLI to build the index
```

---

## 🔧 `requirements.txt`

```txt
fastapi
uvicorn[standard]
pydantic>=2.0
python-multipart
Pillow
numpy
faiss-cpu
torch
torchvision
sentence-transformers
```

> Uses `sentence-transformers` with the CLIP model (`clip-ViT-B-32`) for both text & image embeddings.

---

## 🐳 `Dockerfile`

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# system deps for pillow/faiss
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data

# Expose FastAPI
EXPOSE 8000

# Default cmd (you can override)
CMD ["uvicorn", "app.main:api", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🧻 `.gitignore`

```gitignore
__pycache__/
*.pyc
.env
data/processed/*
!data/raw/.keep
```

Create `data/raw/.keep` (empty file) so the folder exists in git.

---

## 🧠 `app/recommender.py`

```python
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image

import torch
from sentence_transformers import SentenceTransformer

from .utils import load_image_safely

class CLIPRecommender:
    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        index_path: Path = Path("data/processed/embeddings.faiss"),
        meta_path: Path = Path("data/processed/items.json"),
        device: Optional[str] = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index_path = index_path
        self.meta_path = meta_path
        self.index = None
        self.items: List[Dict] = []

        self._load_resources()

    def _load_resources(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.meta_path.exists():
            self.items = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _encode_text(self, text: str) -> np.ndarray:
        emb = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    def _encode_image(self, img: Image.Image) -> np.ndarray:
        emb = self.model.encode([img], convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    def _search(self, query_emb: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Build the index first.")
        D, I = self.index.search(query_emb, k)
        # distances are cosine distances since we normalized; smaller is closer
        results = [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]
        return results

    def recommend_by_text(self, text: str, k: int = 5) -> List[Dict]:
        q = self._encode_text(text)
        hits = self._search(q, k)
        return [self._format_hit(idx, dist) for idx, dist in hits]

    def recommend_by_image(self, image_file, k: int = 5) -> List[Dict]:
        img = load_image_safely(image_file)
        q = self._encode_image(img)
        hits = self._search(q, k)
        return [self._format_hit(idx, dist) for idx, dist in hits]

    def _format_hit(self, idx: int, dist: float) -> Dict:
        item = self.items[idx]
        return {
            "rank": idx,
            "distance": dist,
            "label": item.get("label"),
            "path": item.get("path"),
            "meta": item.get("meta", {})
        }
```

---

## 🧩 `app/ingest.py`

```python
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import faiss
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def scan_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

def label_from_path(p: Path, root: Path) -> str:
    # label == immediate parent folder name (e.g., yoga_pose, playing_guitar)
    rel = p.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "unknown"

def build_index(
    raw_dir: Path = Path("data/raw"),
    out_dir: Path = Path("data/processed"),
    model_name: str = "clip-ViT-B-32",
    batch_size: int = 32
):
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = scan_images(raw_dir)
    if not imgs:
        raise SystemExit(f"No images found in {raw_dir}. Please place dataset under this folder.")

    model = SentenceTransformer(model_name)
    embeddings = []
    items: List[Dict] = []

    # encode in batches
    batch: List[Image.Image] = []
    batch_paths: List[Path] = []

    def flush_batch():
        if not batch: 
            return
        embs = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True)
        embeddings.append(embs.astype("float32"))
        for p in batch_paths:
            items.append({
                "path": str(p.as_posix()),
                "label": label_from_path(p, raw_dir),
                "meta": {}
            })
        batch.clear()
        batch_paths.clear()

    for p in tqdm(imgs, desc="Encoding images"):
        img = Image.open(p).convert("RGB")
        batch.append(img)
        batch_paths.append(p)
        if len(batch) == batch_size:
            flush_batch()
    flush_batch()

    X = np.vstack(embeddings)
    d = X.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine sim via normalized embeddings
    index.add(X)

    faiss.write_index(index, str((out_dir / "embeddings.faiss").as_posix()))
    (out_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved index to {out_dir/'embeddings.faiss'} with {index.ntotal} vectors.")
```

---

## 🧾 `app/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import List, Any

class TextQuery(BaseModel):
    query: str = Field(..., description="Free-form activity query, e.g. 'suggest beginner yoga classes' ")
    k: int = 5

class RecItem(BaseModel):
    rank: int
    distance: float
    label: str
    path: str
    meta: Any = None

class RecResponse(BaseModel):
    results: List[RecItem]
```

---

## 🛠️ `app/utils.py`

```python
from PIL import Image
from io import BytesIO

def load_image_safely(file) -> Image.Image:
    # file can be UploadFile (FastAPI) or file-like
    data = file.file.read() if hasattr(file, "file") else file.read()
    img = Image.open(BytesIO(data)).convert("RGB")
    return img
```

---

## 🌐 `app/main.py`

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .recommender import CLIPRecommender
from .schemas import TextQuery, RecResponse, RecItem

api = FastAPI(title="Multimodal Social Activity Recommendation Engine", version="1.0.0")

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

recsys = CLIPRecommender()

@api.get("/health")
def health():
    return {"status": "ok", "index_loaded": recsys.index is not None, "items": len(recsys.items)}

@api.post("/recommend/text", response_model=RecResponse)
def recommend_text(payload: TextQuery):
    hits = recsys.recommend_by_text(payload.query, k=payload.k)
    return {"results": hits}

@api.post("/recommend/image", response_model=RecResponse)
def recommend_image(file: UploadFile = File(...), k: int = 5):
    hits = recsys.recommend_by_image(file, k=k)
    return {"results": hits}
```

---

## 🧪 `scripts/build_index.py`

```python
from app.ingest import build_index
from pathlib import Path

if __name__ == "__main__":
    # change these if needed
    raw = Path("data/raw")
    out = Path("data/processed")
    build_index(raw_dir=raw, out_dir=out, model_name="clip-ViT-B-32", batch_size=32)
```

- **filters** (e.g., city=“Bangalore”) by enriching `items.json` with metadata and adding query params.
- **re-ranking** (BM25 on event descriptions + CLIP fusion).
- **front-end demo** (minimal React page) to impress recruiters.
::contentReference[oaicite:0]{index=0}
```
