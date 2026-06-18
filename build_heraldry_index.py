"""
Скачивает гербы из Wikimedia Commons и строит офлайн-индекс эмбеддингов.

  python build_heraldry_index.py

Результат: data/heraldry/images/*.png, data/heraldry/index.npz
"""

from __future__ import annotations

import csv
import os
import sys
import time
import urllib.parse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CATALOG = HERE / "heraldry_catalog.csv"
OUT_DIR = HERE / "data" / "heraldry"
IMG_DIR = OUT_DIR / "images"
INDEX_PATH = OUT_DIR / "index.npz"


def _commons_png_url(commons_file: str, width: int = 256) -> str:
    title = commons_file if commons_file.startswith("File:") else f"File:{commons_file}"
    if not title.lower().endswith((".svg", ".png", ".jpg", ".jpeg")):
        title += ".svg"
    fname = title.replace("File:", "")
    return (
        "https://commons.wikimedia.org/wiki/Special:FilePath/"
        + urllib.parse.quote(fname)
        + f"?width={width}"
    )


def _download(url: str, dest: Path, timeout: int = 30) -> bool:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 500:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DIPLOM-heraldry/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 500:
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  skip download {dest.name}: {e}")
        return False


def _embedder():
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

    return embed


def main() -> int:
    if not CATALOG.is_file():
        print(f"Нет каталога: {CATALOG}", file=sys.stderr)
        return 2

    rows: List[dict] = []
    with open(CATALOG, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"Каталог: {len(rows)} записей")
    embed = _embedder()

    ids: List[str] = []
    names: List[str] = []
    kinds: List[str] = []
    lats: List[float] = []
    lons: List[float] = []
    vectors: List[np.ndarray] = []

    for row in rows:
        rid = (row.get("id") or "").strip()
        commons = (row.get("commons_file") or "").strip()
        if not rid or not commons:
            continue

        img_path = IMG_DIR / f"{rid}.png"
        url = _commons_png_url(commons)
        print(f"→ {rid}: {commons}")
        if not _download(url, img_path):
            continue

        try:
            img = Image.open(img_path)
            vec = embed(img)
        except Exception as e:
            print(f"  embed fail: {e}")
            continue

        ids.append(rid)
        names.append(row.get("name", rid))
        kinds.append(row.get("kind", "city"))
        lats.append(float(row.get("lat", 0)))
        lons.append(float(row.get("lon", 0)))
        vectors.append(vec)
        time.sleep(0.15)

    if not vectors:
        print("Не удалось построить ни одного эмбеддинга.", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        INDEX_PATH,
        ids=np.array(ids, dtype=object),
        names=np.array(names, dtype=object),
        kinds=np.array(kinds, dtype=object),
        lats=np.array(lats, dtype=np.float64),
        lons=np.array(lons, dtype=np.float64),
        vectors=np.stack(vectors, axis=0),
    )
    print(f"Готово: {len(ids)} гербов → {INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
