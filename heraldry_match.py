"""
Поиск похожих гербов/флагов по фрагменту изображения (ResNet18 embeddings).
"""

from __future__ import annotations

import base64
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(HERE, "data", "heraldry", "index.npz")

_INDEX: Optional[Dict[str, Any]] = None
_EMBEDDER = None


def _load_index() -> Optional[Dict[str, Any]]:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    if not os.path.isfile(INDEX_PATH):
        return None
    data = np.load(INDEX_PATH, allow_pickle=True)
    _INDEX = {
        "ids": list(data["ids"]),
        "names": list(data["names"]),
        "kinds": list(data["kinds"]),
        "lats": data["lats"],
        "lons": data["lons"],
        "vectors": data["vectors"],
    }
    return _INDEX


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is not None:
        return _EMBEDDER

    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()

    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def embed(img: Image.Image) -> np.ndarray:
        x = transform(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            v = model(x).squeeze(0).numpy()
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        return v.astype(np.float32)

    _EMBEDDER = embed
    return embed


def _image_from_b64(data_b64: str) -> Optional[Image.Image]:
    if not data_b64:
        return None
    try:
        raw = base64.b64decode(data_b64, validate=False)
        return Image.open(io.BytesIO(raw))
    except Exception:
        return None


def match_heraldry_image(img: Image.Image, top_k: int = 5) -> List[Dict[str, Any]]:
    idx = _load_index()
    if idx is None:
        return []

    embed = _get_embedder()
    q = embed(img)
    mat = idx["vectors"]
    scores = mat @ q
    order = np.argsort(-scores)[:top_k]

    out: List[Dict[str, Any]] = []
    for i in order:
        sc = float(scores[i])
        if sc < 0.25:
            continue
        name = str(idx["names"][i])
        kind = str(idx["kinds"][i])
        out.append(
            {
                "name": name,
                "lat": float(idx["lats"][i]),
                "lon": float(idx["lons"][i]),
                "score": round(min(0.99, sc), 3),
                "kind": "heraldry" if kind == "city" else "heraldry_region",
                "detail": f"герб ({kind})",
            }
        )
    return out


def match_heraldry_b64(data_b64: str, top_k: int = 5) -> List[Dict[str, Any]]:
    img = _image_from_b64(data_b64)
    if img is None:
        return []
    return match_heraldry_image(img, top_k=top_k)


def heraldry_index_status() -> Dict[str, Any]:
    idx = _load_index()
    return {
        "heraldry_index": INDEX_PATH,
        "heraldry_loaded": idx is not None,
        "heraldry_count": len(idx["ids"]) if idx else 0,
    }
