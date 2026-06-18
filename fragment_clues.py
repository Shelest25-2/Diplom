"""
Типы фрагментов и авто-определение геоследа (номер / герб / трасса …).
"""

from __future__ import annotations

import re
from typing import List, Tuple

from geo_clues_text import extract_highway_clues, extract_phone_clues, extract_plate_clues

CLUE_TYPE_LABELS: List[Tuple[str, str]] = [
    ("auto", "Авто"),
    ("plate", "Номер"),
    ("heraldry", "Герб"),
    ("flag", "Флаг"),
    ("highway", "Трасса"),
    ("street", "Улица / адрес"),
    ("text", "Текст"),
    ("ignore", "Не анализировать"),
]

CLUE_TYPE_CODES = {code for code, _ in CLUE_TYPE_LABELS}

KIND_PRIORITY = {
    "plate_region": 0,
    "highway": 1,
    "phone_code": 2,
    "heraldry": 3,
    "heraldry_region": 3,
    "street_centroid": 4,
    "house": 4,
    "city": 5,
}


def clue_type_label(code: str) -> str:
    for c, label in CLUE_TYPE_LABELS:
        if c == code:
            return label
    return code or "Авто"


def is_plate_shaped(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    ratio = width / height
    return 2.0 <= ratio <= 6.5


_STREET_HINT_RE = re.compile(r"\b(УЛ|УЛИЦА|ПР[- ]?Т|ПРОСПЕКТ|ПЕР|ПЕРЕУЛОК|ШОССЕ|ПЛОЩАДЬ|ДОМ)\b", re.IGNORECASE)


def _letters_digits_balance(s: str) -> tuple[int, int]:
    letters = sum(1 for ch in s if ch.isalpha())
    digits = sum(1 for ch in s if ch.isdigit())
    return letters, digits


def guess_clue_type_from_text(text: str) -> str:
    if extract_plate_clues(text):
        return "plate"
    if extract_highway_clues(text):
        return "highway"
    if extract_phone_clues(text):
        return "text"
    return "auto"


def effective_clue_type(
    clue_type: str,
    ocr_text: str,
    note: str,
    width: int = 0,
    height: int = 0,
) -> str:
    ct = (clue_type or "auto").strip().lower()
    if ct not in CLUE_TYPE_CODES:
        ct = "auto"
    if ct != "auto":
        return ct

    blob = "\n".join(p for p in (note, ocr_text) if p).strip()
    if blob:
        if _STREET_HINT_RE.search(blob):
            return "street"
        guessed = guess_clue_type_from_text(blob)
        if guessed != "auto":
            return guessed

    if is_plate_shaped(width, height):
        letters, digits = _letters_digits_balance(blob)
        if digits >= 4 and digits >= letters:
            return "plate"
        return "street"

    return "auto"


def allows_heraldry(clue_type: str) -> bool:
    return clue_type in ("auto", "heraldry", "flag")


def allows_text_clues(clue_type: str) -> bool:
    return clue_type not in ("ignore", "heraldry", "flag")


def heraldry_min_score(clue_type: str) -> float:
    if clue_type in ("heraldry", "flag"):
        return 0.35
    return 0.72
