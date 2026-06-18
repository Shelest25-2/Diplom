"""
Текстовые «геоследы» для GeoGuessr-контекста по России:
- код региона на автономерах;
- федеральные трассы (М-/А-/Р-/Е-);
- телефонные коды городов.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class GeoClue:
    kind: str
    name: str
    lat: float
    lon: float
    score: float
    detail: str = ""


def _load_csv_map(path: str, key_col: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (row.get(key_col) or "").strip().upper()
            if key:
                out[key] = row
    return out


_PLATE_MAP: Optional[Dict[str, Dict[str, str]]] = None
_HW_MAP: Optional[Dict[str, Dict[str, str]]] = None


def _plate_map() -> Dict[str, Dict[str, str]]:
    global _PLATE_MAP
    if _PLATE_MAP is None:
        _PLATE_MAP = _load_csv_map(os.path.join(HERE, "plate_regions_ru.csv"), "code")
    return _PLATE_MAP


def _hw_map() -> Dict[str, Dict[str, str]]:
    global _HW_MAP
    if _HW_MAP is None:
        _HW_MAP = _load_csv_map(os.path.join(HERE, "highways_ru.csv"), "code")
    return _HW_MAP


# А123ВС 77 | O777OO177 | Т 123 КМ 54
_PLATE_RE = re.compile(
    r"(?<![A-ZА-Я0-9])"
    r"([A-ZА-Я])\s*(\d{3})\s*([A-ZА-Я]{2})"
    r"\s*(\d{2,3})"
    r"(?![A-ZА-Я0-9])",
    re.IGNORECASE,
)
_PLATE_LOOSE_RE = re.compile(
    r"(?<![0-9])(\d{2,3})(?![0-9])",
)

# М-4, M4, Р 255, Е30
_HW_RE = re.compile(
    r"\b([МАРЕMRAE])\s*[-–]?\s*(\d{1,3})\b",
    re.IGNORECASE,
)

# +7 (383) | 8-383 | (383)
_PHONE_RE = re.compile(
    r"(?:\+7|8)[\s\-()]*(\d{3})|(?:\(\s*(\d{3})\s*\))",
)

# ЧАСТЫЕ городские коды (неполный справочник, легко расширить)
_PHONE_CODES: Dict[str, Tuple[str, float, float]] = {
    "495": ("Москва", 55.7558, 37.6173),
    "499": ("Москва", 55.7558, 37.6173),
    "812": ("Санкт-Петербург", 59.9386, 30.3141),
    "813": ("Санкт-Петербург", 59.9386, 30.3141),
    "383": ("Новосибирск", 55.0084, 82.9357),
    "343": ("Екатеринбург", 56.8389, 60.6057),
    "843": ("Казань", 55.7961, 49.1064),
    "831": ("Нижний Новгород", 56.2965, 43.9361),
    "861": ("Краснодар", 45.0355, 38.9753),
    "863": ("Ростов-на-Дону", 47.2357, 39.7015),
    "846": ("Самара", 53.1959, 50.1002),
    "347": ("Уфа", 54.7388, 55.9721),
    "342": ("Пермь", 58.0105, 56.2502),
    "423": ("Владивосток", 43.1155, 131.8855),
    "4212": ("Хабаровск", 48.4802, 135.0719),
    "3452": ("Тюмень", 57.1530, 65.5343),
    "351": ("Челябинск", 55.1644, 61.4368),
    "3812": ("Омск", 54.9893, 73.3682),
    "473": ("Воронеж", 51.6608, 39.2003),
    "862": ("Сочи", 43.5855, 39.7231),
    "391": ("Красноярск", 56.0153, 92.8932),
    "3952": ("Иркутск", 52.2869, 104.3050),
    "4012": ("Калининград", 54.7104, 20.4522),
    "8152": ("Мурманск", 68.9707, 33.0750),
}


def _latin_to_cyr_hw(letter: str) -> str:
    m = {"M": "М", "A": "А", "R": "Р", "E": "Е"}
    return m.get(letter.upper(), letter.upper())


def extract_plate_clues(text: str) -> List[GeoClue]:
    if not text.strip():
        return []
    t = text.upper().replace("Ё", "Е")
    pmap = _plate_map()
    found: Dict[str, GeoClue] = {}

    for m in _PLATE_RE.finditer(t):
        code = m.group(4)
        row = pmap.get(code)
        if not row:
            continue
        region = row.get("region", code)
        lat = float(row.get("lat", 0))
        lon = float(row.get("lon", 0))
        plate = f"{m.group(1)}{m.group(2)}{m.group(3)} {code}"
        found[code] = GeoClue(
            kind="plate_region",
            name=region,
            lat=lat,
            lon=lon,
            score=0.95,
            detail=f"номер {plate}",
        )

    # Ослабленный режим: отдельно стоящий код региона рядом с «номер/регион/код»
    if not found and re.search(r"регион|номер|код|гос", t, re.I):
        for m in _PLATE_LOOSE_RE.finditer(t):
            code = m.group(1)
            if code not in pmap:
                continue
            row = pmap[code]
            found[code] = GeoClue(
                kind="plate_region",
                name=row.get("region", code),
                lat=float(row.get("lat", 0)),
                lon=float(row.get("lon", 0)),
                score=0.55,
                detail=f"код региона {code}",
            )

    return sorted(found.values(), key=lambda c: c.score, reverse=True)


def extract_highway_clues(text: str) -> List[GeoClue]:
    if not text.strip():
        return []
    t = text.upper().replace("Ё", "Е")
    hmap = _hw_map()
    found: Dict[str, GeoClue] = {}

    for m in _HW_RE.finditer(t):
        letter = _latin_to_cyr_hw(m.group(1))
        num = m.group(2)
        code = f"{letter}-{num}"
        if letter == "М":
            code = f"M-{num}"
        elif letter == "А":
            code = f"A-{num}"
        elif letter == "Р":
            code = f"R-{num}"
        elif letter == "Е":
            code = f"E-{num}"

        row = hmap.get(code)
        if not row:
            continue
        found[code] = GeoClue(
            kind="highway",
            name=row.get("name", code),
            lat=float(row.get("lat", 0)),
            lon=float(row.get("lon", 0)),
            score=0.75,
            detail=f"трасса {code} — {row.get('hint', '')}".strip(),
        )

    return sorted(found.values(), key=lambda c: c.score, reverse=True)


def extract_phone_clues(text: str) -> List[GeoClue]:
    if not text.strip():
        return []
    found: Dict[str, GeoClue] = {}
    for m in _PHONE_RE.finditer(text):
        code = (m.group(1) or m.group(2) or "").strip()
        if not code:
            continue
        # Сначала 4-значные, потом 3-значные
        for width in (4, 3):
            sub = code[:width]
            hit = _PHONE_CODES.get(sub)
            if hit:
                city, lat, lon = hit
                found[sub] = GeoClue(
                    kind="phone_code",
                    name=city,
                    lat=lat,
                    lon=lon,
                    score=0.65,
                    detail=f"тел. код {sub}",
                )
                break
    return sorted(found.values(), key=lambda c: c.score, reverse=True)


def extract_all_text_clues(text: str) -> List[GeoClue]:
    clues: List[GeoClue] = []
    clues.extend(extract_plate_clues(text))
    clues.extend(extract_highway_clues(text))
    clues.extend(extract_phone_clues(text))
    clues.sort(key=lambda c: c.score, reverse=True)
    return clues


def clues_to_topk(clues: List[GeoClue], limit: int = 8) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen: set[str] = set()
    for c in clues[:limit]:
        key = f"{c.kind}:{c.name}"
        if key in seen:
            continue
        seen.add(key)
        label = c.name
        if c.detail:
            label = f"{c.name} ({c.detail})"
        out.append(
            {
                "name": label,
                "lat": c.lat,
                "lon": c.lon,
                "score": c.score,
                "kind": c.kind,
            }
        )
    return out
