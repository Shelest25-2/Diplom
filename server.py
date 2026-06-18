import csv
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="DIPLOM Locate Server", version="0.2.0")
HERE = os.path.dirname(os.path.abspath(__file__))
GEO_DB_PATH = os.environ.get("DIPLOM_GEO_DB", os.path.join(HERE, "geo.db"))

try:
    from rapidfuzz.fuzz import partial_ratio  # type: ignore
except Exception:
    partial_ratio = None

class OverlayRect(BaseModel):
    x: int
    y: int
    w: int
    h: int


class OverlayFragment(BaseModel):
    id: str
    created_at_ms: int
    rect: OverlayRect
    note: str = ""
    ocr_text: str = ""
    image_b64: str = ""
    clue_type: str = "auto"


class LocateRequest(BaseModel):
    schema_: str = Field(default="diplom.overlay.v1", alias="schema")
    created_at_ms: int
    manual_text: str = ""
    fragments: List[OverlayFragment] = Field(default_factory=list)
    mode: str = "geoguess"  # geoguess | address


@dataclass(frozen=True)
class City:
    name: str
    lat: float
    lon: float
    aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Street:
    city: str
    name: str
    aliases: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Intersection:
    city: str
    street_a: str
    street_b: str
    lat: float
    lon: float


@dataclass(frozen=True)
class House:
    city: str
    street: str
    house: str
    lat: float
    lon: float


# Minimal seed list (expand later from CSV/GeoNames/OpenStreetMap extracts).
CITY_DB_FALLBACK: List[City] = [
    City("Москва", 55.7558, 37.6173, ("мск", "moscow")),
    City("Санкт‑Петербург", 59.9386, 30.3141, ("питер", "спб", "saint petersburg", "st petersburg")),
    City("Новосибирск", 55.0084, 82.9357, ("академгородок", "novosibirsk")),
    City("Екатеринбург", 56.8389, 60.6057, ("екб", "yekaterinburg")),
    City("Казань", 55.7961, 49.1064, ("kazan",)),
    City("Нижний Новгород", 56.2965, 43.9361, ("нн", "nizhny novgorod")),
    City("Краснодар", 45.0355, 38.9753, ("krasnodar",)),
    City("Сочи", 43.5855, 39.7231, ("sochi",)),
    City("Ростов‑на‑Дону", 47.2357, 39.7015, ("ростов", "rostov-on-don")),
    City("Самара", 53.1959, 50.1002, ("samara",)),
    City("Уфа", 54.7388, 55.9721, ("ufa",)),
    City("Пермь", 58.0105, 56.2502, ("perm",)),
    City("Владивосток", 43.1155, 131.8855, ("vladivostok",)),
    City("Хабаровск", 48.4802, 135.0719, ("khabarovsk",)),
    City("Тюмень", 57.1530, 65.5343, ("tyumen",)),
    City("Челябинск", 55.1644, 61.4368, ("chelyabinsk",)),
    City("Омск", 54.9893, 73.3682, ("omsk",)),
    City("Воронеж", 51.6608, 39.2003, ("voronezh",)),
]

def load_city_db() -> List[City]:
    """
    Loads local city DB from ru_cities.csv (offline).
    CSV columns:
      name,lat,lon,aliases
    where aliases is optional and uses '|' separator.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "ru_cities.csv")
    if not os.path.isfile(path):
        return CITY_DB_FALLBACK

    out: List[City] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            lat_s = (row.get("lat") or "").strip()
            lon_s = (row.get("lon") or "").strip()
            aliases_s = (row.get("aliases") or "").strip()
            if not name or not lat_s or not lon_s:
                continue
            try:
                lat = float(lat_s.replace(",", "."))
                lon = float(lon_s.replace(",", "."))
            except Exception:
                continue
            aliases = tuple(a.strip() for a in aliases_s.split("|") if a.strip()) if aliases_s else ()
            out.append(City(name=name, lat=lat, lon=lon, aliases=aliases))

    return out or CITY_DB_FALLBACK


CITY_DB: List[City] = load_city_db()

MAJOR_CITY_NAMES = {c.name for c in CITY_DB}

CITY_ALIAS_TO_NAME: Dict[str, str] = {}

def _read_csv_dicts(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_streets() -> List[Street]:
    """
    streets.csv columns:
      city,street,aliases
    aliases: optional, '|' separated.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "streets.csv")
    out: List[Street] = []
    for row in _read_csv_dicts(path):
        city = (row.get("city") or "").strip()
        street = (row.get("street") or "").strip()
        aliases_s = (row.get("aliases") or "").strip()
        if not city or not street:
            continue
        aliases = tuple(a.strip() for a in aliases_s.split("|") if a.strip()) if aliases_s else ()
        out.append(Street(city=city, name=street, aliases=aliases))
    return out


def load_intersections() -> List[Intersection]:
    """
    intersections.csv columns:
      city,street_a,street_b,lat,lon
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "intersections.csv")
    out: List[Intersection] = []
    for row in _read_csv_dicts(path):
        city = (row.get("city") or "").strip()
        a = (row.get("street_a") or "").strip()
        b = (row.get("street_b") or "").strip()
        lat_s = (row.get("lat") or "").strip()
        lon_s = (row.get("lon") or "").strip()
        if not city or not a or not b or not lat_s or not lon_s:
            continue
        try:
            lat = float(lat_s.replace(",", "."))
            lon = float(lon_s.replace(",", "."))
        except Exception:
            continue
        out.append(Intersection(city=city, street_a=a, street_b=b, lat=lat, lon=lon))
    return out


def load_houses() -> List[House]:
    """
    houses.csv columns:
      city,street,house,lat,lon
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "houses.csv")
    out: List[House] = []
    for row in _read_csv_dicts(path):
        city = (row.get("city") or "").strip()
        street = (row.get("street") or "").strip()
        house = (row.get("house") or "").strip()
        lat_s = (row.get("lat") or "").strip()
        lon_s = (row.get("lon") or "").strip()
        if not city or not street or not house or not lat_s or not lon_s:
            continue
        try:
            lat = float(lat_s.replace(",", "."))
            lon = float(lon_s.replace(",", "."))
        except Exception:
            continue
        out.append(House(city=city, street=street, house=house, lat=lat, lon=lon))
    return out


STREET_DB: List[Street] = load_streets()
INTERSECTION_DB: List[Intersection] = load_intersections()
HOUSE_DB: List[House] = load_houses()


CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
LATIN_RE = re.compile(r"[A-Za-z]")
TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)


def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    return s


for _city in CITY_DB:
    for _alias in (_city.name, *_city.aliases):
        _key = normalize(_alias)
        if _key:
            CITY_ALIAS_TO_NAME[_key] = _city.name

CITY_NAME_NORMS = {normalize(c.name) for c in CITY_DB}

REPEAT_RE = re.compile(r"(.)\1{2,}", re.UNICODE)

_STREET_MARKER_TOKENS = frozenset(
    {
        "ул",
        "улица",
        "проспект",
        "пр-т",
        "переулок",
        "шоссе",
        "бульвар",
        "набережная",
        "площадь",
    }
)


def street_marker_count(text: str) -> int:
    """How many typed street markers appear (ул, улица, проспект, …)."""
    norm = normalize_ocr_noise(text)
    rx = len(STREET_TYPE_RE.findall(norm))
    tk = sum(1 for t in tokenize(text) if normalize(t) in _STREET_MARKER_TOKENS)
    return max(rx, tk)


def street_like_name_tokens(text: str) -> List[str]:
    """
    Ordered street-looking tokens (length >= 4), excluding city names/aliases.
    Used when typed markers appear twice but regex splitting failed (encoding/OCR quirks).
    """
    out: List[str] = []
    for t in tokenize(text):
        if len(t) < 4:
            continue
        tn = normalize(t)
        if tn in CITY_ALIAS_TO_NAME or tn in CITY_NAME_NORMS:
            continue
        ns = normalize_street_name(t)
        if not ns or len(ns) < 2:
            continue
        if ns in CITY_ALIAS_TO_NAME or ns in CITY_NAME_NORMS:
            continue
        out.append(ns)
    return list(dict.fromkeys(out))


def street_queries_for_pairing(text: str) -> List[str]:
    prim = extract_street_candidates(text)
    if len(prim) >= 2:
        return prim[:8]
    if street_marker_count(text) >= 2:
        tok = street_like_name_tokens(text)
        if len(tok) >= 2:
            return tok[:8]
    return prim


def street_name_pairs_from_text(text: str) -> List[Tuple[str, str]]:
    """
    Имена улиц (name_norm-ready), все неупорядоченные пары для поиска пересечений.

    Если в тексте >2 улиц («Новосельская Степная Тихвинская», три строки),
    нужно проверять все пары — только первые две часто не пересекаются.
    """
    names_raw = street_queries_for_pairing(text)
    names = []
    for n in names_raw:
        nn = normalize_street_name(n)
        if nn and len(nn) >= 2:
            names.append(nn)
    names = list(dict.fromkeys(names))
    if len(names) < 2:
        return []
    if len(names) == 2:
        a, b = names[0], names[1]
        return [(a, b)] if a != b else []
    max_names = 6
    names = names[:max_names]
    pairs: List[Tuple[str, str]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if names[i] != names[j]:
                pairs.append((names[i], names[j]))
    return pairs[:20]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Приблизительное расстояние между точками на сфере (км)."""
    from math import asin, cos, radians, sin, sqrt

    rlat1, rlon1, rlat2, rlon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    h = sin(dlat / 2) ** 2 + cos(rlat1) * cos(rlat2) * sin(dlon / 2) ** 2
    return 6371.0 * (2 * asin(min(1.0, sqrt(h))))


def _strip_leading_city_aliases(chunk: str) -> str:
    """\"новосиб 1-й экскаваторный\" - \"1-й экскаваторный\"."""
    parts = chunk.split()
    while parts:
        key = normalize(parts[0])
        if key in CITY_ALIAS_TO_NAME or key in CITY_NAME_NORMS:
            parts.pop(0)
            continue
        break
    return " ".join(parts).strip()


def _strip_trailing_city_aliases(chunk: str) -> str:
    """\"вертковская 36/1 новосиб\" - \"вертковская 36/1\"."""
    parts = chunk.split()
    while parts:
        key = normalize(parts[-1])
        if key in CITY_ALIAS_TO_NAME or key in CITY_NAME_NORMS:
            parts.pop()
            continue
        break
    return " ".join(parts).strip()


def _strip_city_like_street_chunks(chunks: List[str]) -> List[str]:
    out: List[str] = []
    for c in chunks:
        cn = normalize(c)
        if not cn:
            continue
        if cn in CITY_ALIAS_TO_NAME or cn in CITY_NAME_NORMS:
            continue
        out.append(c)
    return out


def normalize_ocr_noise(s: str) -> str:
    s = normalize(s)
    # «ул. Вертковская, 36» — запятая перед номером мешает токенам и хвосту дома
    s = re.sub(r",\s*(\d+[a-zа-я]?)\b", r" \1", s, flags=re.IGNORECASE | re.UNICODE)
    s = re.sub(r",\s*", " ", s)
    # OCR often glues digits to words: "народная12345" -> split for tokenize/pairing.
    s = re.sub(r"([А-Яа-яЁё])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([А-Яа-яЁё])", r"\1 \2", s)
    # Keep two repeats (helps words like "ссср" not needed, but prevents over-collapsing).
    s = REPEAT_RE.sub(r"\1\1", s)
    return s


def detect_script(s: str) -> str:
    has_cyr = bool(CYRILLIC_RE.search(s))
    has_lat = bool(LATIN_RE.search(s))
    if has_cyr and has_lat:
        return "mixed"
    if has_cyr:
        return "cyrillic"
    if has_lat:
        return "latin"
    return "unknown"


def tokenize(s: str) -> List[str]:
    s = normalize_ocr_noise(s)
    raw = TOKEN_RE.findall(s)
    merged: List[str] = []
    ord_suffix = frozenset({"й", "я", "е", "ё"})
    i = 0
    while i < len(raw):
        if (
            i + 1 < len(raw)
            and raw[i].isdigit()
            and len(raw[i]) <= 3
            and normalize(raw[i + 1]) in ord_suffix
        ):
            merged.append(f"{raw[i]}-{raw[i + 1]}")
            i += 2
            continue
        merged.append(raw[i])
        i += 1
    return [t for t in merged if len(t) >= 2]


def text_from_request(req: LocateRequest) -> str:
    parts = [req.manual_text]
    for f in req.fragments:
        if f.note:
            parts.append(f.note)
        if f.ocr_text:
            parts.append(f.ocr_text)
    return "\n".join(p for p in parts if p)


def geo_db_paths_to_try() -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for p in (GEO_DB_PATH, os.path.join(os.getcwd(), "geo.db"), os.path.join(HERE, "geo.db")):
        ap = os.path.normpath(os.path.abspath(p))
        if ap not in seen:
            seen.add(ap)
            out.append(ap)
    return out


def resolve_geo_db_path() -> Optional[str]:
    for path in geo_db_paths_to_try():
        if os.path.isfile(path):
            return path
    return None


def open_geo_db() -> Optional[sqlite3.Connection]:
    path = resolve_geo_db_path()
    if path is None:
        return None
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def _city_key_in_text(key: str, text_norm: str, tokens: List[str]) -> bool:
    if not key:
        return False
    if len(key) <= 3:
        tok_set = {normalize(t) for t in tokens}
        return key in tok_set
    return key in text_norm


def _fuzzy_allows_city_token(token: str, key: str, pr: float) -> bool:
    if pr < 88.0:
        return False
    tn, kn = normalize(token), normalize(key)
    if not tn or not kn:
        return False
    if kn.startswith(tn) or tn.startswith(kn):
        return True
    if abs(len(kn) - len(tn)) <= 1:
        return pr >= 88.0
    return pr >= 94.0


def score_city(city: City, text_norm: str, tokens: List[str]) -> float:
    name_n = normalize(city.name)
    keys = [name_n, *[normalize(a) for a in city.aliases]]

    score = 0.0
    for key in keys:
        if not key:
            continue
        if _city_key_in_text(key, text_norm, tokens):
            score += 5.0

        key_tokens = set(tokenize(key))
        if key_tokens:
            overlap = len(set(tokens) & key_tokens)
            if overlap:
                score += 1.5 * overlap

        if partial_ratio is not None and len(key) >= 4 and len(text_norm) >= 4:
            try:
                pr = float(partial_ratio(key, text_norm))  # 0..100
                if pr >= 90:
                    score += (pr - 89) * 0.08  # up to ~0.9
                elif pr >= 80:
                    score += (pr - 79) * 0.03  # up to ~0.3
            except Exception:
                pass

        if partial_ratio is not None and len(key) >= 4 and tokens:
            best_pr = 0.0
            for t in tokens:
                if len(t) < 4:
                    continue
                try:
                    pr_t = float(partial_ratio(key, t))
                except Exception:
                    continue
                if not _fuzzy_allows_city_token(t, key, pr_t):
                    continue
                best_pr = max(best_pr, pr_t)
            if best_pr >= 88:
                score += (best_pr - 87) * 0.05  # up to ~0.65
            elif best_pr >= 80:
                score += (best_pr - 79) * 0.015  # up to ~0.15

    if name_n in tokens:
        score += 2.0

    return score


STREET_TYPE_RE = re.compile(
    r"\b(?:ул|улица|пр-?т|проспект|переулок|шоссе|бул|бульвар|наб|набережная|пл|площадь)\b\.?",
    re.IGNORECASE | re.UNICODE,
)
STREET_MARKER_RE = re.compile(
    r"(?:^|\b)(?:ул|улица|пр-?т|проспект|переулок|шоссе|бул|бульвар|наб|набережная|пл|площадь)\.?\s+(.+)$",
    re.IGNORECASE | re.UNICODE,
)
# Дом: 10, 10а, 10/2, 36/1 (дробный номер в РФ)
HOUSE_RE = re.compile(
    r"\b(?:д|дом)\s*\.?\s*(\d+(?:/\d+)?[a-zа-я]?)\b",
    re.IGNORECASE | re.UNICODE,
)
HOUSE_FALLBACK_RE = re.compile(r"\b(\d+[a-zа-я]?)\b", re.IGNORECASE | re.UNICODE)
HOUSE_COMPOUND_RE = re.compile(
    r"\b(\d{1,4}/\d{1,4}[a-zа-я]?)\b",
    re.IGNORECASE | re.UNICODE,
)
# Срез номера с конца названия улицы (в т.ч. 36/1)
_TRAILING_HOUSE_TAIL_RE = re.compile(
    r"(?<=\s)(?:\d{1,4}/\d{1,4}[a-zа-я]?|\d+[a-zа-я]?)\s*$",
    re.IGNORECASE | re.UNICODE,
)
# "1-й Примерный переулок" — не номер дома
ORDINAL_STREET_TOKEN_RE = re.compile(
    r"\b\d{1,3}\s*-\s*[йяеё]\b",
    re.IGNORECASE | re.UNICODE,
)
ORDINAL_STREET_NAME_RE = re.compile(
    r"^(\d{1,3})\s*-\s*([йяеё])\s+(.+)$",
    re.UNICODE | re.IGNORECASE,
)
ORDINAL_TOKEN_ONLY_RE = re.compile(r"^\d{1,3}\s*-\s*[йяеё]$", re.UNICODE | re.IGNORECASE)


def normalize_street_name(raw: str) -> str:
    s = normalize_ocr_noise(raw)
    s = STREET_TYPE_RE.sub(" ", s)
    s = s.replace("  ", " ").strip()
    return s


def street_type_hint_from_query(text: str) -> Optional[str]:
    """
    Явный класс дороги из запроса («ул …», «переулок …»), чтобы не путать
    одноимённые улицу и переулок в БД.
    """
    n = normalize_ocr_noise(text)
    if re.search(r"\bпереулок\b", n, re.IGNORECASE):
        return "pereulok"
    if re.search(r"\bпроспект\b|\bпр-т\b", n, re.IGNORECASE):
        return "prospekt"
    if re.search(r"\bплощадь\b", n, re.IGNORECASE):
        return "ploshchad"
    if re.search(r"\bул\.?\b|\bулица\b", n, re.IGNORECASE):
        return "ulitsa"
    return None


def extract_street_candidates(text: str) -> List[str]:
    norm = normalize_ocr_noise(text)
    out: List[str] = []

    def _strip_house_tail(s: str) -> str:
        s = re.sub(
            r"\b(?:д|дом)\s*\.?\s*\d+(?:/\d+)?[a-zа-я]?\b.*$",
            "",
            s,
            flags=re.IGNORECASE | re.UNICODE,
        )
        # Хвост «… 36/1» или «… 10» (не съедаем «1-й» в начале строки)
        s = _TRAILING_HOUSE_TAIL_RE.sub("", s)
        return s.strip()

    def _emit_from_piece(piece: str, acc: List[str]) -> None:
        piece = piece.strip()
        piece = _strip_trailing_city_aliases(piece)
        piece = _strip_leading_city_aliases(piece)
        piece = _strip_house_tail(piece)
        if not piece:
            return
        if STREET_TYPE_RE.search(piece):
            for sub in STREET_TYPE_RE.split(piece):
                _emit_from_piece(sub, acc)
            return
        sn = normalize_street_name(piece)
        if sn and len(sn) >= 2:
            acc.append(sn)

    has_street_type = bool(STREET_TYPE_RE.search(norm))
    if has_street_type:
        cleaned: List[str] = []
        for raw in STREET_TYPE_RE.split(norm):
            _emit_from_piece(raw, cleaned)
        cleaned = list(dict.fromkeys(cleaned))
        candidates = _strip_city_like_street_chunks(cleaned)
        if len(candidates) >= 2:
            return candidates
        if len(candidates) == 1:
            return candidates

    m = STREET_MARKER_RE.search(norm)
    if m:
        phrase = (m.group(1) or "").strip()
        cleaned_marker: List[str] = []
        _emit_from_piece(phrase, cleaned_marker)
        cleaned_marker = list(dict.fromkeys(cleaned_marker))
        candidates_m = _strip_city_like_street_chunks(cleaned_marker)
        if len(candidates_m) >= 2:
            return candidates_m
        if len(candidates_m) == 1:
            return candidates_m

    toks = tokenize(text)
    skip_next = False
    for i, t in enumerate(toks):
        if skip_next:
            skip_next = False
            continue
        tn = normalize(t)
        if tn in CITY_ALIAS_TO_NAME or tn in CITY_NAME_NORMS:
            continue
        if tn in _STREET_MARKER_TOKENS:
            continue
        if ORDINAL_TOKEN_ONLY_RE.match(t.strip()) and i + 1 < len(toks):
            nxt = toks[i + 1]
            nn = normalize(nxt)
            if (
                nn not in _STREET_MARKER_TOKENS
                and nn not in CITY_ALIAS_TO_NAME
                and nn not in CITY_NAME_NORMS
            ):
                merged = normalize_street_name(f"{t} {nxt}")
                if merged and len(merged) >= 2:
                    out.append(merged)
                skip_next = True
                continue
        if len(t) >= 4:
            out.append(normalize_street_name(t))
    return _strip_city_like_street_chunks(list(dict.fromkeys([x for x in out if x])))


def extract_house_number(text: str, *, allow_bare_number: bool = True) -> Optional[str]:
    norm = normalize_ocr_noise(text)
    norm_h = ORDINAL_STREET_TOKEN_RE.sub(" ", norm)
    m = HOUSE_RE.search(norm_h)
    if m:
        return (m.group(1) or "").strip()
    # Составной номер 10/2, 36/1 — до голых цифр (иначе из «36/1» берётся «1»)
    compounds = list(HOUSE_COMPOUND_RE.finditer(norm_h))
    if compounds:
        return compounds[-1].group(1).strip()
    if not allow_bare_number:
        return None
    norm_fb = HOUSE_COMPOUND_RE.sub(" ", norm_h)
    all_nums = HOUSE_FALLBACK_RE.findall(norm_fb)
    if not all_nums:
        return None
    return (all_nums[-1] or "").strip()


def house_number_to_int(h: str) -> Optional[int]:
    m = re.search(r"\d+", normalize_ocr_noise(h or ""))
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def house_norm_lookup_keys(raw: str) -> List[str]:
    """
    В OSM часто «10/2» или «10к2» — пробуем оба при точном совпадении house_norm.
    """
    hn = normalize_ocr_noise(raw).replace(" ", "")
    if not hn:
        return []
    keys: List[str] = [hn]
    m_slash = re.match(r"^(\d+)/(\d+)([a-zа-я]?)$", hn, re.IGNORECASE)
    if m_slash:
        keys.append(f"{m_slash.group(1)}к{m_slash.group(2)}{m_slash.group(3) or ''}")
    m_k = re.match(r"^(\d+)к(\d+)([a-zа-я]?)$", hn, re.IGNORECASE)
    if m_k:
        keys.append(f"{m_k.group(1)}/{m_k.group(2)}{m_k.group(3) or ''}")
    return list(dict.fromkeys(keys))


def fuzzy_best(query: str, choices: List[str]) -> Tuple[float, Optional[str]]:
    if partial_ratio is None or not query or not choices:
        return 0.0, None
    best_s = 0.0
    best_c: Optional[str] = None
    for c in choices:
        try:
            s = float(partial_ratio(query, normalize_ocr_noise(c)))
        except Exception:
            continue
        if s > best_s:
            best_s, best_c = s, c
    return best_s, best_c


def resolve_house(text: str) -> Optional[Dict[str, Any]]:
    if street_marker_count(text) >= 2:
        return None
    house_no = extract_house_number(text, allow_bare_number=True)
    if not house_no:
        return None

    street_queries_pre = extract_street_candidates(text)
    street_queries = street_queries_pre
    if not street_queries:
        return None

    streets = list({h.street for h in HOUSE_DB})
    best = None
    best_score = 0.0
    for q in street_queries:
        s, street = fuzzy_best(q, streets)
        if street and s > best_score:
            best_score = s
            best = street

    if not best or best_score < 78:
        return None

    hn = normalize_ocr_noise(house_no).replace(" ", "")
    matches = [h for h in HOUSE_DB if normalize_street_name(h.street) == normalize_street_name(best) and normalize_ocr_noise(h.house).replace(" ", "") == hn]
    if not matches:
        return None

    h = matches[0]
    return {
        "kind": "house",
        "city": h.city,
        "street": h.street,
        "house": h.house,
        "lat": h.lat,
        "lon": h.lon,
        "score": round(best_score / 100.0, 4),
    }


def resolve_house_sqlite(text: str) -> Optional[Dict[str, Any]]:
    if street_marker_count(text) >= 2:
        return None
    house_no = extract_house_number(text, allow_bare_number=True)
    if not house_no:
        return None

    street_queries_pre = extract_street_candidates(text)

    tokens = tokenize(text)
    if not tokens:
        return None

    city_hint_name: Optional[str] = None
    hint_token_keys: set[str] = set()
    for t in tokens:
        t_key = normalize(t)
        if t_key in CITY_ALIAS_TO_NAME:
            city_hint_name = CITY_ALIAS_TO_NAME[t_key]
            hint_token_keys.add(t_key)
            break

    exclude_token_keys = hint_token_keys

    street_queries = street_queries_pre
    street_queries = [
        q for q in street_queries
        if not (len(tokenize(q)) == 1 and normalize(q) in exclude_token_keys)
    ]

    if not street_queries:
        return None

    con = open_geo_db()
    if con is None:
        return None

    try:
        house_lookup_keys = house_norm_lookup_keys(house_no)
        house_num_target = house_number_to_int(house_no)
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0

        city_id_hint: Optional[int] = None
        if city_hint_name:
            city_norm = normalize(city_hint_name)
            row = con.execute(
                "SELECT city_id FROM cities WHERE name_norm = ? LIMIT 1",
                (city_norm,),
            ).fetchone()
            if row:
                city_id_hint = int(row["city_id"])

        for q in street_queries[:8]:
            qn = normalize_street_name(q)
            if not qn:
                continue
            qlike = f"%{qn[: max(2, min(16, len(qn)))]}%"
            q_tokens = [t for t in tokenize(qn) if len(t) >= 3]

            exact_sql = """
                SELECT s.street_id, s.name AS street_name, s.name_norm, c.city_id, c.name AS city_name
                FROM streets s
                JOIN cities c ON c.city_id = s.city_id
                WHERE s.name_norm = ?
            """
            exact_params: List[Any] = [qn]
            if city_id_hint is not None:
                exact_sql += " AND c.city_id = ? "
                exact_params.append(city_id_hint)
            exact_sql += " LIMIT 40"
            rows = con.execute(exact_sql, tuple(exact_params)).fetchall()

            if not rows:
                base_sql = """
                SELECT s.street_id, s.name AS street_name, s.name_norm, c.city_id, c.name AS city_name
                FROM streets s
                JOIN cities c ON c.city_id = s.city_id
                WHERE s.name_norm LIKE ?
                """
                params: List[Any] = [qlike]
                if city_id_hint is not None:
                    base_sql += " AND c.city_id = ? "
                    params.append(city_id_hint)
                base_sql += " LIMIT 120"
                rows = con.execute(base_sql, tuple(params)).fetchall()
            if not rows:
                continue

            road_hint = street_type_hint_from_query(text)

            for r in rows:
                sname_low = (r["street_name"] or "").lower()
                if road_hint == "ulitsa" and "переулок" in sname_low:
                    continue
                if road_hint == "pereulok" and "улица" in sname_low and "переулок" not in sname_low:
                    continue
                if road_hint == "prospekt" and "площадь" in sname_low and "проспект" not in sname_low:
                    continue
                if road_hint == "ploshchad" and "проспект" in sname_low and "площадь" not in sname_low:
                    continue

                street_norm = r["name_norm"] or ""
                s_tokens = [t for t in tokenize(street_norm) if len(t) >= 3]
                if len(q_tokens) >= 2:
                    overlap = len(set(q_tokens) & set(s_tokens))
                    if overlap < 2:
                        continue

                if partial_ratio is not None:
                    try:
                        s_score = float(partial_ratio(qn, street_norm))
                    except Exception:
                        s_score = 0.0
                else:
                    s_score = 100.0 if qn in street_norm or street_norm in qn else 0.0

                if s_score < 72:
                    continue

                hrow = None
                matched_house_norm = ""
                for hk in house_lookup_keys:
                    hrow = con.execute(
                        """
                        SELECT h.lat, h.lon, h.house
                        FROM houses h
                        WHERE h.city_id = ? AND h.street_id = ? AND h.house_norm = ?
                        LIMIT 1
                        """,
                        (r["city_id"], r["street_id"], hk),
                    ).fetchone()
                    if hrow:
                        matched_house_norm = hk
                        break
                if not hrow:
                    approx_rows = con.execute(
                        """
                        SELECT h.lat, h.lon, h.house
                        FROM houses h
                        WHERE h.city_id = ? AND h.street_id = ?
                        LIMIT 500
                        """,
                        (r["city_id"], r["street_id"]),
                    ).fetchall()
                    if not approx_rows:
                        continue

                    best_approx = None
                    best_dist = 10**9
                    for ar in approx_rows:
                        hn = ar["house"] or ""
                        if house_num_target is not None:
                            hn_num = house_number_to_int(hn)
                            if hn_num is None:
                                continue
                            dist = abs(hn_num - house_num_target)
                        else:
                            dist = 0
                        if dist < best_dist:
                            best_dist = dist
                            best_approx = ar

                    if best_approx is None:
                        continue

                    penalty = 0.15 if best_dist <= 2 else 0.25 if best_dist <= 10 else 0.35
                    final_score = round(max(0.0, min(1.0, s_score / 100.0 - penalty)), 4)
                    if final_score > best_score:
                        best_score = final_score
                        best = {
                            "kind": "house_approx",
                            "city": r["city_name"],
                            "street": r["street_name"],
                            "house": best_approx["house"],
                            "lat": float(best_approx["lat"]),
                            "lon": float(best_approx["lon"]),
                            "score": final_score,
                            "source": "sqlite",
                            "requested_house": house_no,
                            "exact_house_match": False,
                        }
                    continue

                final_score = round(min(1.0, s_score / 100.0), 4)
                if final_score > best_score:
                    best_score = final_score
                    best = {
                        "kind": "house",
                        "city": r["city_name"],
                        "street": r["street_name"],
                        "house": hrow["house"],
                        "lat": float(hrow["lat"]),
                        "lon": float(hrow["lon"]),
                        "score": final_score,
                        "source": "sqlite",
                    }

        return best
    finally:
        con.close()


def _city_id_from_hint(con: sqlite3.Connection, city_hint_name: Optional[str]) -> Optional[int]:
    if not city_hint_name:
        return None
    row = con.execute(
        "SELECT city_id FROM cities WHERE name_norm = ? LIMIT 1",
        (normalize(city_hint_name),),
    ).fetchone()
    return int(row["city_id"]) if row else None


def resolve_intersection_sqlite(text: str) -> Optional[List[Dict[str, Any]]]:
    pairs_to_try = street_name_pairs_from_text(text)
    if not pairs_to_try:
        return None

    tokens = tokenize(text)
    city_hint_name: Optional[str] = None
    for t in tokens:
        t_key = normalize(t)
        if t_key in CITY_ALIAS_TO_NAME:
            city_hint_name = CITY_ALIAS_TO_NAME[t_key]
            break

    con = open_geo_db()
    if con is None:
        return None

    try:
        n_inter = con.execute("SELECT COUNT(*) AS c FROM intersections").fetchone()
        if not n_inter or int(n_inter["c"]) == 0:
            return None

        city_id_hint = _city_id_from_hint(con, city_hint_name)
        hits: List[Dict[str, Any]] = []

        def _query_pair(a_norm: str, b_norm: str) -> None:
            sql = """
            SELECT i.lat, i.lon, c.name AS city,
                   sa.name AS street_a, sb.name AS street_b
            FROM intersections i
            JOIN streets sa ON sa.street_id = i.street_a_id
            JOIN streets sb ON sb.street_id = i.street_b_id
            JOIN cities c ON c.city_id = i.city_id
            WHERE sa.name_norm = ? AND sb.name_norm = ?
            """
            params: List[Any] = [a_norm, b_norm]
            if city_id_hint is not None:
                sql += " AND i.city_id = ? "
                params.append(city_id_hint)
            for r in con.execute(sql, tuple(params)).fetchall():
                hits.append(
                    {
                        "kind": "intersection",
                        "city": r["city"],
                        "street_a": r["street_a"],
                        "street_b": r["street_b"],
                        "lat": float(r["lat"]),
                        "lon": float(r["lon"]),
                        "score": 0.92,
                        "source": "sqlite",
                    }
                )

        for q1, q2 in pairs_to_try:
            if not q1 or not q2:
                continue
            _query_pair(q1, q2)
            _query_pair(q2, q1)

        if not hits:
            return None
        seen_pairs: set[Tuple[str, str, str]] = set()
        uniq: List[Dict[str, Any]] = []
        for h in hits:
            sk = (str(h["city"]),) + tuple(sorted([str(h["street_a"]), str(h["street_b"])]))
            if sk in seen_pairs:
                continue
            seen_pairs.add(sk)
            uniq.append(h)

        def _inter_sort_key(x: Dict[str, Any]) -> Tuple[int, float]:
            major = 1 if x["city"] in MAJOR_CITY_NAMES else 0
            return (major, float(x["score"]))

        uniq.sort(key=_inter_sort_key, reverse=True)
        return uniq
    finally:
        con.close()


def resolve_street_pair_centroids_sqlite(text: str) -> Optional[List[Dict[str, Any]]]:
    pairs_to_try = street_name_pairs_from_text(text)
    if not pairs_to_try:
        return None

    tokens = tokenize(text)
    city_hint_name: Optional[str] = None
    for t in tokens:
        t_key = normalize(t)
        if t_key in CITY_ALIAS_TO_NAME:
            city_hint_name = CITY_ALIAS_TO_NAME[t_key]
            break

    con = open_geo_db()
    if con is None:
        return None

    try:
        city_id_hint = _city_id_from_hint(con, city_hint_name)

        def _centroid(city_id: int, street_id: int) -> Optional[Tuple[float, float, int]]:
            row = con.execute(
                """
                SELECT AVG(lat) AS alat, AVG(lon) AS alon, COUNT(*) AS n
                FROM houses
                WHERE city_id = ? AND street_id = ?
                """,
                (city_id, street_id),
            ).fetchone()
            if not row or int(row["n"]) == 0:
                return None
            return float(row["alat"]), float(row["alon"]), int(row["n"])

        sql_base = """
            SELECT c.city_id, c.name AS city, s1.street_id AS sid1, s1.name AS n1,
                   s2.street_id AS sid2, s2.name AS n2
            FROM streets s1
            JOIN streets s2 ON s1.city_id = s2.city_id AND s1.street_id < s2.street_id
            JOIN cities c ON c.city_id = s1.city_id
            WHERE s1.name_norm = ? AND s2.name_norm = ?
        """
        hits: List[Dict[str, Any]] = []

        for q1, q2 in pairs_to_try:
            if not q1 or not q2 or q1 == q2:
                continue
            params_list: List[List[Any]] = [[q1, q2], [q2, q1]]

            for params in params_list:
                sql = sql_base
                p: List[Any] = list(params)
                if city_id_hint is not None:
                    sql += " AND c.city_id = ? "
                    p.append(city_id_hint)
                for r in con.execute(sql, tuple(p)).fetchall():
                    cid = int(r["city_id"])
                    c1 = _centroid(cid, int(r["sid1"]))
                    c2 = _centroid(cid, int(r["sid2"]))
                    if c1 is None or c2 is None:
                        continue
                    sep_km = round(_haversine_km(c1[0], c1[1], c2[0], c2[1]), 3)
                    lat = (c1[0] + c2[0]) / 2.0
                    lon = (c1[1] + c2[1]) / 2.0
                    nmin = min(c1[2], c2[2])
                    score = round(max(0.45, 0.82 - 1.0 / (4.0 + nmin)), 4)
                    hits.append(
                        {
                            "kind": "intersection_approx",
                            "city": r["city"],
                            "street_a": r["n1"],
                            "street_b": r["n2"],
                            "lat": lat,
                            "lon": lon,
                            "score": score,
                            "source": "sqlite_centroid",
                            "houses_street_a": c1[2],
                            "houses_street_b": c2[2],
                            "centroid_sep_km": sep_km,
                        }
                    )

        if not hits:
            return None

        seen_pairs: set[Tuple[str, str, str]] = set()
        uniq: List[Dict[str, Any]] = []
        for h in hits:
            sk = (str(h["city"]),) + tuple(sorted([str(h["street_a"]), str(h["street_b"])]))
            if sk in seen_pairs:
                continue
            seen_pairs.add(sk)
            uniq.append(h)

        def _pair_sort_key(x: Dict[str, Any]) -> Tuple[int, float, float, int, int]:
            """
            При одинаковых name_norm в разных городах: ru_cities.csv - ближе центроиды - score - дома.
            """
            major = 1 if x["city"] in MAJOR_CITY_NAMES else 0
            sep = float(x.get("centroid_sep_km", 9999.0))
            ha = int(x.get("houses_street_a", 0))
            hb = int(x.get("houses_street_b", 0))
            return (major, -sep, float(x["score"]), min(ha, hb), ha + hb)

        uniq.sort(key=_pair_sort_key, reverse=True)
        return uniq
    finally:
        con.close()


def _build_centroids_from_ordinal_fallback(
    con: sqlite3.Connection,
    q_norm: str,
    city_id_hint: Optional[int],
) -> List[Dict[str, Any]]:
    m = ORDINAL_STREET_NAME_RE.match(q_norm.strip())
    if not m:
        return []
    q_ord = int(m.group(1))
    base = (m.group(3) or "").strip().lower()
    if len(base) < 2:
        return []
    like_pat = f"%{base}%"
    if city_id_hint is not None:
        cands = con.execute(
            """
            SELECT c.city_id, c.name AS city, s.street_id, s.name AS street_name, s.name_norm
            FROM streets s
            JOIN cities c ON c.city_id = s.city_id
            WHERE c.city_id = ? AND s.name_norm LIKE ?
            LIMIT 160
            """,
            (city_id_hint, like_pat),
        ).fetchall()
    else:
        cands = con.execute(
            """
            SELECT c.city_id, c.name AS city, s.street_id, s.name AS street_name, s.name_norm
            FROM streets s
            JOIN cities c ON c.city_id = s.city_id
            WHERE s.name_norm LIKE ?
            LIMIT 240
            """,
            (like_pat,),
        ).fetchall()

    ranked: List[Tuple[int, float, sqlite3.Row]] = []
    for r in cands:
        nn = (r["name_norm"] or "").strip().lower()
        m2 = ORDINAL_STREET_NAME_RE.match(nn)
        if not m2:
            continue
        row_base = (m2.group(3) or "").strip().lower()
        if row_base != base:
            continue
        row_ord = int(m2.group(1))
        dist = abs(row_ord - q_ord)
        major = 0.2 if r["city"] in MAJOR_CITY_NAMES else 0.0
        ranked.append((dist, -major, r))
    ranked.sort(key=lambda x: (x[0], x[1]))

    out: List[Dict[str, Any]] = []
    for dist, _maj, r in ranked[:8]:
        agg = con.execute(
            """
            SELECT AVG(lat) AS alat, AVG(lon) AS alon, COUNT(*) AS n
            FROM houses
            WHERE city_id = ? AND street_id = ?
            """,
            (int(r["city_id"]), int(r["street_id"])),
        ).fetchone()
        if not agg or int(agg["n"]) == 0:
            continue
        n = int(agg["n"])
        score = round(max(0.22, 0.52 - 0.06 * float(dist) - 1.0 / (3.0 + float(n))), 4)
        out.append(
            {
                "kind": "street_approx",
                "city": r["city"],
                "street": r["street_name"],
                "lat": float(agg["alat"]),
                "lon": float(agg["alon"]),
                "score": score,
                "source": "sqlite_ordinal_fallback",
                "houses": n,
                "requested_street_norm": q_norm,
                "ordinal_distance": dist,
                "matched_name_norm": r["name_norm"],
            }
        )
    return out


def resolve_street_centroids_sqlite(text: str) -> Optional[List[Dict[str, Any]]]:
    street_queries = extract_street_candidates(text)
    if len(street_queries) != 1:
        return None

    q = normalize_street_name(street_queries[0])
    if not q or len(q) < 2:
        return None

    tokens = tokenize(text)
    city_hint_name: Optional[str] = None
    for t in tokens:
        t_key = normalize(t)
        if t_key in CITY_ALIAS_TO_NAME:
            city_hint_name = CITY_ALIAS_TO_NAME[t_key]
            break

    con = open_geo_db()
    if con is None:
        return None

    try:
        city_id_hint = _city_id_from_hint(con, city_hint_name)
        sql = """
            SELECT c.city_id, c.name AS city, s.street_id, s.name AS street_name
            FROM streets s
            JOIN cities c ON c.city_id = s.city_id
            WHERE s.name_norm = ?
        """
        params: List[Any] = [q]
        if city_id_hint is not None:
            sql += " AND c.city_id = ? "
            params.append(city_id_hint)
        rows = con.execute(sql, tuple(params)).fetchall()

        hits: List[Dict[str, Any]] = []
        for r in rows:
            agg = con.execute(
                """
                SELECT AVG(lat) AS alat, AVG(lon) AS alon, COUNT(*) AS n
                FROM houses
                WHERE city_id = ? AND street_id = ?
                """,
                (int(r["city_id"]), int(r["street_id"])),
            ).fetchone()
            if not agg or int(agg["n"]) == 0:
                continue
            n = int(agg["n"])
            score = round(max(0.38, 0.68 - 1.0 / (3.0 + float(n))), 4)
            hits.append(
                {
                    "kind": "street_centroid",
                    "city": r["city"],
                    "street": r["street_name"],
                    "lat": float(agg["alat"]),
                    "lon": float(agg["alon"]),
                    "score": score,
                    "source": "sqlite_centroid",
                    "houses": n,
                }
            )

        if not hits:
            hits = _build_centroids_from_ordinal_fallback(con, q, city_id_hint)

        if not hits:
            return None

        def _street_sort_key(x: Dict[str, Any]) -> Tuple[float, int]:
            major_boost = 0.1 if x["city"] in MAJOR_CITY_NAMES else 0.0
            return (float(x["score"]) + major_boost, int(x.get("houses", 0)))

        hits.sort(key=_street_sort_key, reverse=True)
        return hits[:12]
    finally:
        con.close()


def resolve_intersection(text: str) -> Optional[Dict[str, Any]]:
    street_queries = street_queries_for_pairing(text)
    if len(street_queries) < 2:
        return None

    known = list({normalize_street_name(i.street_a) for i in INTERSECTION_DB} | {normalize_street_name(i.street_b) for i in INTERSECTION_DB})
    if not known:
        return None

    matched: List[Tuple[str, float]] = []
    for q in street_queries[:8]:
        s, st = fuzzy_best(q, known)
        if st and s >= 75:
            matched.append((st, s))

    if len(matched) < 2:
        return None

    matched.sort(key=lambda x: x[1], reverse=True)
    streets: List[str] = []
    for st, _s in matched:
        if st not in streets:
            streets.append(st)
        if len(streets) == 2:
            break
    if len(streets) < 2:
        return None

    a, b = streets[0], streets[1]
    for inter in INTERSECTION_DB:
        ia = normalize_street_name(inter.street_a)
        ib = normalize_street_name(inter.street_b)
        if {ia, ib} == {a, b}:
            return {
                "kind": "intersection",
                "city": inter.city,
                "street_a": inter.street_a,
                "street_b": inter.street_b,
                "lat": inter.lat,
                "lon": inter.lon,
                "score": 0.8,
            }
    return None


@app.get("/health")
def health() -> Dict[str, Any]:
    resolved_db = resolve_geo_db_path()
    db_stats: Dict[str, Any] = {
        "sqlite_enabled": False,
        "geo_db_path": resolved_db,
        "geo_db_candidates": geo_db_paths_to_try(),
    }
    con = open_geo_db()
    if con is not None:
        try:
            db_stats["sqlite_enabled"] = True
            for table_name in ("cities", "streets", "houses", "intersections"):
                try:
                    row = con.execute(f"SELECT COUNT(*) AS cnt FROM {table_name}").fetchone()
                    db_stats[f"{table_name}_db"] = int(row["cnt"] if row else 0)
                except Exception:
                    db_stats[f"{table_name}_db"] = None
        finally:
            con.close()

    heraldry_stats: Dict[str, Any] = {"heraldry_loaded": False, "heraldry_count": 0}
    try:
        from heraldry_match import heraldry_index_status

        heraldry_stats = heraldry_index_status()
    except Exception:
        pass

    return {
        "ok": True,
        "version": app.version,
        "cities_loaded": len(CITY_DB),
        "fuzzy": partial_ratio is not None,
        "streets_loaded": len(STREET_DB),
        "intersections_loaded": len(INTERSECTION_DB),
        "houses_loaded": len(HOUSE_DB),
        **db_stats,
        **heraldry_stats,
    }


def _intersection_response_payload(
    script: str,
    tokens: List[str],
    items: List[Dict[str, Any]],
    evidence_key: str,
) -> Dict[str, Any]:
    topk: List[Dict[str, Any]] = []
    for x in items[:8]:
        topk.append(
            {
                "name": f"{x['city']}: {x['street_a']} ∩ {x['street_b']}",
                "lat": x["lat"],
                "lon": x["lon"],
                "score": float(x["score"]),
                "kind": str(x.get("kind", "intersection")),
            }
        )
    return {
        "schema": "diplom.locate_response.v1",
        "script": script,
        "tokens": tokens[:80],
        "topk": topk,
        "best": topk[0] if topk else None,
        "evidence": {evidence_key: items},
    }


def _street_centroid_response_payload(
    script: str,
    tokens: List[str],
    items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    topk: List[Dict[str, Any]] = []
    for x in items[:8]:
        topk.append(
            {
                "name": f"{x['city']}, {x['street']}",
                "lat": x["lat"],
                "lon": x["lon"],
                "score": float(x["score"]),
                "kind": str(x.get("kind", "street_centroid")),
            }
        )
    return {
        "schema": "diplom.locate_response.v1",
        "script": script,
        "tokens": tokens[:80],
        "topk": topk,
        "best": topk[0] if topk else None,
        "evidence": {"streets": items},
    }


def _merge_topk(items: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for it in sorted(items, key=lambda x: float(x.get("score", 0)), reverse=True):
        key = f"{it.get('kind')}:{it.get('name')}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(it)
        if len(merged) >= limit:
            break
    return merged


def _sort_geoguess_candidates(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from fragment_clues import KIND_PRIORITY

    def key(it: Dict[str, Any]) -> Tuple[int, float]:
        kind = str(it.get("kind") or "")
        pr = KIND_PRIORITY.get(kind, 50)
        return (pr, -float(it.get("score", 0)))

    return sorted(items, key=key)


def resolve_geoguess(req: LocateRequest) -> Optional[Dict[str, Any]]:
    """GeoGuessr-сигналы: номера, трассы, гербы/флаги — с учётом clue_type фрагмента."""
    from fragment_clues import (
        allows_heraldry,
        allows_text_clues,
        clue_type_label,
        effective_clue_type,
        heraldry_min_score,
    )
    from geo_clues_text import clues_to_topk, extract_all_text_clues

    try:
        from heraldry_match import match_heraldry_b64
    except Exception:
        match_heraldry_b64 = None  # type: ignore

    raw_text = text_from_request(req)
    tokens = tokenize(raw_text)
    script = detect_script(raw_text)

    candidates: List[Dict[str, Any]] = []
    evidence: Dict[str, Any] = {}
    fragment_analysis: List[Dict[str, Any]] = []
    has_plate = False
    has_street_text = any(
        effective_clue_type(f.clue_type, f.ocr_text, f.note, f.rect.w, f.rect.h) in ("street", "text", "highway")
        and bool((f.ocr_text or "").strip())
        for f in req.fragments
    )

    for frag in req.fragments:
        eff = effective_clue_type(
            frag.clue_type,
            frag.ocr_text,
            frag.note,
            frag.rect.w,
            frag.rect.h,
        )
        analysis: Dict[str, Any] = {
            "id": frag.id,
            "clue_type": eff,
            "clue_type_label": clue_type_label(eff),
            "detected_kind": "",
            "detected_label": "",
            "score": 0.0,
        }

        if eff == "ignore":
            fragment_analysis.append(analysis)
            continue

        frag_text = "\n".join(p for p in (frag.note, frag.ocr_text) if p).strip()

        if allows_text_clues(eff):
            frag_clues = extract_all_text_clues(frag_text)
            if frag_clues:
                best = frag_clues[0]
                analysis["detected_kind"] = best.kind
                analysis["detected_label"] = best.name
                analysis["score"] = best.score
                if best.kind == "plate_region":
                    has_plate = True
                for c in frag_clues:
                    candidates.append(
                        {
                            "name": f"{c.name} ({c.detail})" if c.detail else c.name,
                            "lat": c.lat,
                            "lon": c.lon,
                            "score": c.score,
                            "kind": c.kind,
                            "fragment_id": frag.id,
                        }
                    )

        if (
            match_heraldry_b64 is not None
            and allows_heraldry(eff)
            and eff not in ("plate", "highway", "street", "text")
            and not (has_street_text and eff == "auto")
        ):
            b64 = (frag.image_b64 or "").strip()
            if b64 and not (has_plate and eff == "auto"):
                min_sc = heraldry_min_score(eff)
                hits = match_heraldry_b64(b64, top_k=3)
                good = [h for h in hits if float(h.get("score", 0)) >= min_sc]
                if good and not analysis["detected_kind"]:
                    h0 = good[0]
                    analysis["detected_kind"] = str(h0.get("kind", "heraldry"))
                    analysis["detected_label"] = str(h0.get("name", ""))
                    analysis["score"] = float(h0.get("score", 0))
                for h in good:
                    hc = dict(h)
                    hc["fragment_id"] = frag.id
                    candidates.append(hc)

        fragment_analysis.append(analysis)

    # Ручной текст / фрагменты без явной разметки
    if req.manual_text.strip():
        manual_clues = extract_all_text_clues(req.manual_text)
        if manual_clues:
            if any(c.kind == "plate_region" for c in manual_clues):
                has_plate = True
            evidence["manual_clues"] = [
                {"kind": c.kind, "name": c.name, "detail": c.detail, "score": c.score} for c in manual_clues
            ]
            candidates.extend(clues_to_topk(manual_clues))

    if has_plate:
        candidates = [c for c in candidates if str(c.get("kind")) != "heraldry" and str(c.get("kind")) != "heraldry_region"]

    if not candidates:
        if fragment_analysis:
            return {
                "schema": "diplom.locate_response.v1",
                "script": script,
                "tokens": tokens[:80],
                "mode": "geoguess",
                "topk": [],
                "best": None,
                "evidence": {"fragment_analysis": fragment_analysis},
                "fragment_analysis": fragment_analysis,
            }
        return None

    ranked = _sort_geoguess_candidates(candidates)
    topk = _merge_topk(ranked)
    evidence["fragment_analysis"] = fragment_analysis

    return {
        "schema": "diplom.locate_response.v1",
        "script": script,
        "tokens": tokens[:80],
        "mode": "geoguess",
        "topk": topk,
        "best": topk[0] if topk else None,
        "evidence": evidence,
        "fragment_analysis": fragment_analysis,
    }


@app.post("/locate")
def locate(req: LocateRequest) -> Dict[str, Any]:
    mode = (req.mode or "geoguess").strip().lower()
    if mode != "address":
        geo = resolve_geoguess(req)
        # Пустой geoguess (например, только «ул. X / ул. Y») — не блокировать адресный поиск.
        if geo is not None and geo.get("topk"):
            return geo

    raw_text = text_from_request(req)
    text_norm = normalize_ocr_noise(raw_text)
    tokens = tokenize(raw_text)

    script = detect_script(raw_text)

    # 1) House-level (if address-like text found)
    house = resolve_house_sqlite(raw_text) or resolve_house(raw_text)
    if house is not None:
        return {
            "schema": "diplom.locate_response.v1",
            "script": script,
            "tokens": tokens[:80],
            "topk": [
                {
                    "name": f"{house['city']}, {house['street']}, {house['house']}",
                    "lat": house["lat"],
                    "lon": house["lon"],
                    "score": house["score"],
                    "kind": "house",
                }
            ],
            "best": {
                "name": f"{house['city']}, {house['street']}, {house['house']}",
                "lat": house["lat"],
                "lon": house["lon"],
                "score": house["score"],
                "kind": "house",
            },
            "evidence": {"house": house},
        }

    # 2) Intersections: OSM table (exact) -> CSV demo -> two-street centroid from houses
    inter_sql = resolve_intersection_sqlite(raw_text)
    if inter_sql:
        return _intersection_response_payload(script, tokens, inter_sql, "intersections")

    inter = resolve_intersection(raw_text)
    if inter is not None:
        return _intersection_response_payload(script, tokens, [inter], "intersection")

    pair_sql = resolve_street_pair_centroids_sqlite(raw_text)
    if pair_sql:
        return _intersection_response_payload(script, tokens, pair_sql, "street_pairs")

    street_centroids = resolve_street_centroids_sqlite(raw_text)
    if street_centroids:
        return _street_centroid_response_payload(script, tokens, street_centroids)

    scored: List[Tuple[float, City]] = []
    for city in CITY_DB:
        s = score_city(city, text_norm, tokens)
        if s > 0:
            scored.append((s, city))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:5]

    resp = {
        "schema": "diplom.locate_response.v1",
        "script": script,
        "tokens": tokens[:80],
        "topk": [
            {
                "name": c.name,
                "lat": c.lat,
                "lon": c.lon,
                "score": float(s),
                "kind": "city",
            }
            for s, c in top
        ],
        "best": None,
    }

    if top:
        s, c = top[0]
        resp["best"] = {"name": c.name, "lat": c.lat, "lon": c.lon, "score": float(s), "kind": "city"}

    return resp

