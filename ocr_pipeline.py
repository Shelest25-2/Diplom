"""
OCR pipeline for screen fragments.

Движки:
  - EasyOCR — таблички / улицы / общий текст на «живых» скринах (рекомендуется)
  - Tesseract — номера (whitelist), запасной вариант

Установка EasyOCR: python -m pip install easyocr
"""
from __future__ import annotations

import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

try:
    import pytesseract  # type: ignore
    from pytesseract import Output  # type: ignore
except Exception:
    pytesseract = None  # type: ignore
    Output = None  # type: ignore

_EASYOCR_READER = None
_EASYOCR_INIT_FAILED = False
_EASYOCR_INIT_MESSAGE = ""
_LAST_OCR_ENGINE = "none"

_EASYOCR_REQUIRED_FILES = ("craft_mlt_25k.pth", "cyrillic_g2.pth")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _env_lang() -> str:
    return (os.environ.get("DIPLOM_OCR_LANG", "rus+eng") or "rus+eng").strip()


def _env_psm_list() -> List[int]:
    raw = (os.environ.get("DIPLOM_OCR_PSM_LIST", "6,11,3") or "6,11,3").strip()
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    return out or [6, 11, 3]


def pil_from_rgb(pil: Image.Image) -> Image.Image:
    if pil.mode != "RGB":
        return pil.convert("RGB")
    return pil


def maybe_upscale(pil: Image.Image, min_side: Optional[int] = None) -> Image.Image:
    if min_side is None:
        min_side = _env_int("DIPLOM_OCR_MIN_SIDE", 640)
    w, h = pil.size
    m = max(w, h)
    if m >= min_side:
        return pil
    scale = float(min_side) / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return pil.resize((nw, nh), Image.Resampling.LANCZOS)


def enhance_for_ocr(pil: Image.Image) -> Image.Image:
    img = pil_from_rgb(pil)
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = ImageEnhance.Contrast(gray).enhance(1.25)
    gray = ImageEnhance.Sharpness(gray).enhance(1.15)
    return gray


def _pil_to_bgr_uint8(pil: Image.Image) -> "np.ndarray":
    rgb = pil_from_rgb(pil)
    arr = np.asarray(rgb)
    return arr[:, :, ::-1].copy()


def _bgr_to_pil_l(bgr: "np.ndarray") -> Image.Image:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return Image.fromarray(gray, mode="L")


def deskew_small_angle(pil: Image.Image, max_angle: float = 12.0) -> Image.Image:
    if os.environ.get("DIPLOM_OCR_SKIP_DESKEW", "").strip() in ("1", "true", "yes"):
        return pil
    if cv2 is None or np is None:
        return pil
    try:
        bgr = _pil_to_bgr_uint8(pil)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if coords.shape[0] < 20:
            return pil
        angle = float(cv2.minAreaRect(coords)[-1])
        if angle < -45:
            angle = 90.0 + angle
        else:
            angle = -angle
        if abs(angle) > max_angle or abs(angle) < 0.15:
            return pil
        h, w = bgr.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            bgr,
            m,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return _bgr_to_pil_l(rotated).convert("RGB")
    except Exception:
        return pil


def rotate_via_osd(pil: Image.Image) -> Image.Image:
    if pytesseract is None or Output is None:
        return pil
    if os.environ.get("DIPLOM_OCR_SKIP_OSD", "").strip() in ("1", "true", "yes"):
        return pil
    try:
        info = pytesseract.image_to_osd(pil, lang="osd", output_type=Output.DICT)
    except Exception:
        return pil
    try:
        rot = int(info.get("rotate", 0)) % 360
    except Exception:
        return pil
    if rot == 0:
        return pil
    # StackOverflow / pytesseract convention: rotate by (360 - rot) to upright.
    return pil.rotate(360 - rot, expand=True, fillcolor=(255, 255, 255))


def _mean_word_confidence(pil: Image.Image, lang: str, psm: int, extra_cfg: str = "") -> Tuple[float, str]:
    if pytesseract is None or Output is None:
        return 0.0, ""
    extra_cfg = (extra_cfg or "").strip()
    cfg = f"--oem 3 --psm {psm} -c user_defined_dpi=300 {extra_cfg}".strip()
    try:
        data = pytesseract.image_to_data(
            pil,
            lang=lang,
            config=cfg,
            output_type=Output.DICT,
        )
    except Exception:
        return 0.0, ""
    confs: List[int] = []
    for c in data.get("conf", []):
        try:
            v = int(float(c))
        except Exception:
            continue
        if v > 0:
            confs.append(v)
    mean_conf = float(sum(confs)) / float(len(confs)) if confs else 0.0
    try:
        text = pytesseract.image_to_string(pil, lang=lang, config=cfg) or ""
    except Exception:
        text = ""
    text = text.strip()
    return mean_conf, text


def _score_text(t: str) -> int:
    if not t:
        return 0
    s = t.strip()
    if len(s) <= 2:
        return 0
    letters = sum(1 for ch in s if ch.isalpha())
    digits = sum(1 for ch in s if ch.isdigit())
    spaces = sum(1 for ch in s if ch.isspace())
    words = len([w for w in re.split(r"\s+", s) if w])
    # Reward words and letters, but keep some value for digits.
    score = letters * 5 + digits * 2 + len(s) + words * 12 + min(6, spaces) * 2
    # Strong penalty for ultra-short garbage like "ГЕ" / "II".
    if letters <= 2 and words <= 1 and len(s) <= 4:
        score -= 50
    return max(0, score)


_RU_PLATE_ALLOWED = "ABEKMHOPCTYXАВЕКМНОРСТУХ0123456789"
_RU_PLATE_RE = re.compile(r"([ABEKMHOPCTYXАВЕКМНОРСТУХ])\s*(\d{3})\s*([ABEKMHOPCTYXАВЕКМНОРСТУХ]{2})\s*(\d{2,3})")


def _latin_cyr_equiv(s: str) -> str:
    # Map both directions to reduce OCR alphabet mix.
    m = {
        "A": "А",
        "B": "В",
        "E": "Е",
        "K": "К",
        "M": "М",
        "H": "Н",
        "O": "О",
        "P": "Р",
        "C": "С",
        "T": "Т",
        "Y": "У",
        "X": "Х",
        "А": "А",
        "В": "В",
        "Е": "Е",
        "К": "К",
        "М": "М",
        "Н": "Н",
        "О": "О",
        "Р": "Р",
        "С": "С",
        "Т": "Т",
        "У": "У",
        "Х": "Х",
    }
    out = []
    for ch in s:
        out.append(m.get(ch, ch))
    return "".join(out)


_PLATE_REGION_CODES: Optional[set[str]] = None


def _load_plate_region_codes() -> set[str]:
    global _PLATE_REGION_CODES
    if _PLATE_REGION_CODES is not None:
        return _PLATE_REGION_CODES
    codes: set[str] = set()
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "plate_regions_ru.csv")
        if os.path.isfile(path):
            import csv

            with open(path, newline="", encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    c = (row.get("code") or "").strip()
                    if c:
                        codes.add(c)
    except Exception:
        pass
    _PLATE_REGION_CODES = codes
    return codes


def _fix_region_code(code: str) -> str:
    codes = _load_plate_region_codes()
    if not codes or code in codes:
        return code
    if "3" in code:
        cand = code.replace("3", "8", 1)
        if cand in codes:
            return cand
    if "8" in code:
        cand = code.replace("8", "3", 1)
        if cand in codes:
            return cand
    return code


def _normalize_plate_text(s: str) -> str:
    s = (s or "").upper().replace("Ё", "Е")
    s = _latin_cyr_equiv(s)
    s = re.sub(r"[^A-ZА-Я0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_best_plate(text: str) -> str:
    t = _normalize_plate_text(text)
    hits: List[str] = []
    for m in _RU_PLATE_RE.finditer(t):
        a = m.group(1)
        d = m.group(2)
        bb = m.group(3)
        reg = _fix_region_code(m.group(4))
        hits.append(f"{a}{d}{bb}{reg}")
    if not hits:
        return ""
    counts: Dict[str, int] = {}
    for h in hits:
        counts[h] = counts.get(h, 0) + 1
    return sorted(counts.items(), key=lambda kv: (kv[1], len(kv[0])), reverse=True)[0][0]


def enhance_for_plate(pil: Image.Image) -> Image.Image:
    img = pil_from_rgb(pil)
    img = maybe_upscale(img, min_side=_env_int("DIPLOM_OCR_PLATE_MIN_SIDE", 900))
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.7)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    if cv2 is not None and np is not None:
        try:
            arr = np.asarray(gray)
            # Adaptive threshold helps with highlights / shadows on plates.
            thr = cv2.adaptiveThreshold(
                arr,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                9,
            )
            # Clean small noise.
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
            gray = Image.fromarray(thr, mode="L")
        except Exception:
            pass
    return gray


_RU_TEXT_WHITELIST = (
    "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    "0123456789"
    " /.-,"
)


def _env_street_lang() -> str:
    return (os.environ.get("DIPLOM_OCR_STREET_LANG", "rus") or "rus").strip()


def _env_ocr_engine() -> str:
    v = (os.environ.get("DIPLOM_OCR_ENGINE", "auto") or "auto").strip().lower()
    if v not in ("auto", "easyocr", "tesseract"):
        return "auto"
    return v


def last_ocr_engine() -> str:
    return _LAST_OCR_ENGINE


def _set_ocr_engine(name: str) -> None:
    global _LAST_OCR_ENGINE
    _LAST_OCR_ENGINE = name


def easyocr_model_dir() -> str:
    base = os.environ.get("EASYOCR_MODULE_PATH") or os.environ.get("MODULE_PATH")
    if base:
        return os.path.join(base, "model")
    return os.path.join(os.path.expanduser("~"), ".EasyOCR", "model")


def easyocr_models_ready() -> bool:
    model_dir = easyocr_model_dir()
    for name in _EASYOCR_REQUIRED_FILES:
        if not os.path.isfile(os.path.join(model_dir, name)):
            return False
    return True


def easyocr_package_installed() -> bool:
    try:
        import easyocr  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def easyocr_available() -> bool:
    """Пакет установлен и модели уже на диске (без скачивания из GUI)."""
    if _EASYOCR_INIT_FAILED:
        return False
    return easyocr_package_installed() and easyocr_models_ready()


def easyocr_status_message() -> str:
    if _EASYOCR_INIT_MESSAGE:
        return _EASYOCR_INIT_MESSAGE
    if not easyocr_package_installed():
        return "EasyOCR не установлен (pip install easyocr)."
    if not easyocr_models_ready():
        return (
            "EasyOCR: модели не скачаны. В терминале: python download_easyocr_models.py "
            "(если зависло на «Downloading…» — это норма без моделей; GUI использует Tesseract)."
        )
    return "EasyOCR готов."


def _get_easyocr_reader():
    global _EASYOCR_READER, _EASYOCR_INIT_FAILED, _EASYOCR_INIT_MESSAGE
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    if _EASYOCR_INIT_FAILED:
        return None
    if not easyocr_package_installed():
        _EASYOCR_INIT_MESSAGE = "Пакет easyocr не найден."
        _EASYOCR_INIT_FAILED = True
        return None
    if not easyocr_models_ready():
        allow = os.environ.get("DIPLOM_OCR_ALLOW_DOWNLOAD", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if not allow:
            _EASYOCR_INIT_MESSAGE = (
                "Модели EasyOCR не найдены. Скачать: python download_easyocr_models.py"
            )
            return None

    try:
        import easyocr  # type: ignore
        import threading

        langs = [
            x.strip()
            for x in (os.environ.get("DIPLOM_OCR_EASYOCR_LANGS", "ru,en") or "ru,en").split(",")
            if x.strip()
        ]
        gpu = os.environ.get("DIPLOM_OCR_EASYOCR_GPU", "0").strip().lower() in ("1", "true", "yes")
        allow_dl = os.environ.get("DIPLOM_OCR_ALLOW_DOWNLOAD", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )

        holder: List[Any] = [None]
        err_holder: List[Optional[BaseException]] = [None]

        def _init() -> None:
            try:
                holder[0] = easyocr.Reader(
                    langs,
                    gpu=gpu,
                    verbose=False,
                    download_enabled=allow_dl,
                    model_storage_directory=easyocr_model_dir(),
                )
            except BaseException as e:
                err_holder[0] = e

        t = threading.Thread(target=_init, daemon=True)
        t.start()
        timeout_s = _env_int("DIPLOM_OCR_EASYOCR_INIT_TIMEOUT", 120)
        t.join(timeout=timeout_s)
        if t.is_alive():
            _EASYOCR_INIT_FAILED = True
            _EASYOCR_INIT_MESSAGE = (
                f"Таймаут загрузки EasyOCR ({timeout_s} с). "
                "Скачать модели: python download_easyocr_models.py"
            )
            return None
        if err_holder[0] is not None:
            raise err_holder[0]

        _EASYOCR_READER = holder[0]
        return _EASYOCR_READER
    except Exception as e:
        _EASYOCR_INIT_FAILED = True
        _EASYOCR_INIT_MESSAGE = f"EasyOCR: {e!r}"
        return None


def _group_easyocr_items(items: List[Tuple[float, str, float]], y_tol: float = 0.08) -> List[str]:
    """Склеить слова EasyOCR в строки по вертикали."""
    if not items:
        return []
    items = sorted(items, key=lambda x: x[0])
    y_span = max(items[-1][0] - items[0][0], 1.0)
    tol = max(12.0, y_span * y_tol)
    lines: List[str] = []
    bucket_y: List[float] = []
    bucket_txt: List[str] = []
    for y, text, _conf in items:
        if not bucket_txt or abs(y - bucket_y[-1]) <= tol:
            bucket_txt.append(text)
            bucket_y.append(y)
        else:
            lines.append(" ".join(bucket_txt))
            bucket_txt = [text]
            bucket_y = [y]
    if bucket_txt:
        lines.append(" ".join(bucket_txt))
    return lines


def _street_line_sort_key(line: str) -> Tuple[int, str]:
    """Порядок строк на вертикальной табличке: номер - улица - название."""
    s = line.strip()
    if re.fullmatch(r"\d{1,4}(/[\w\d]+)?", s):
        return (0, s)
    if re.search(r"\bул", s, re.IGNORECASE):
        return (1, s)
    return (2, s)


def _merge_street_line_candidates(candidates: List[str]) -> str:
    """Собрать лучшие строки из нескольких прогонов EasyOCR (разные препроцессинги)."""
    if not candidates:
        return ""
    line_best: Dict[str, str] = {}
    for cand in candidates:
        for raw_line in re.split(r"[\n]+", cand):
            line = _post_process_text(raw_line)
            if not line:
                continue
            key = re.sub(r"\s+", "", line.lower())
            prev = line_best.get(key)
            if prev is None or _score_text(line) > _score_text(prev):
                line_best[key] = line
            # Длиннее похожая строка (Казбекск → Казбекская).
            for k, v in list(line_best.items()):
                if k != key and (key.startswith(k) or k.startswith(key)):
                    better = line if len(line) >= len(v) else v
                    del line_best[k]
                    nk = re.sub(r"\s+", "", better.lower())
                    line_best[nk] = better
    if not line_best:
        return max((_post_process_text(c) for c in candidates), key=_score_text, default="")
    ordered = sorted(line_best.values(), key=_street_line_sort_key)
    if len(ordered) >= 2:
        return "\n".join(ordered)
    return ordered[0]


def _blue_panel_mask(hsv: "np.ndarray") -> "np.ndarray":
    """Маска синей адресной таблички (в т.ч. в тени)."""
    if cv2 is None:
        return hsv[:, :, 0] * 0
    m1 = cv2.inRange(hsv, (90, 50, 40), (135, 255, 255))
    m2 = cv2.inRange(hsv, (85, 25, 25), (140, 255, 255))
    m3 = cv2.inRange(hsv, (95, 15, 15), (130, 255, 210))
    mask = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return mask


def _score_panel_contour(cnt: Any, mask: "np.ndarray", w0: int, h0: int) -> float:
    if cv2 is None:
        return -1.0
    area = float(cv2.contourArea(cnt))
    if area <= 0:
        return -1.0
    x, y, bw, bh = cv2.boundingRect(cnt)
    if bw < 25 or bh < 18:
        return -1.0
    roi = mask[y : y + bh, x : x + bw]
    if roi.size == 0:
        return -1.0
    fill = float((roi > 0).sum()) / float(roi.size)
    if fill < 0.22:
        return -1.0
    ar = bw / max(bh, 1)
    ar_bonus = 1.25 if 0.2 < ar < 4.5 else 0.75
    min_area = max(0.012 * w0 * h0, 600.0)
    if area < min_area:
        return -1.0
    return area * fill * ar_bonus


def _easyocr_input_variants(pil: Image.Image) -> List["np.ndarray"]:
    """Несколько RGB-вариантов для EasyOCR: оригинал, обрезка, белый на синем, бинаризация."""
    img = pil_from_rgb(pil)
    img = maybe_upscale(img, min_side=_env_int("DIPLOM_OCR_TEXT_MIN_SIDE", 1200))
    cropped = _crop_sign_panel(img)
    cropped = maybe_upscale(cropped, min_side=_env_int("DIPLOM_OCR_TEXT_MIN_SIDE", 1200))

    variants: List["np.ndarray"] = []
    seen: set[int] = set()

    def _add(arr: "np.ndarray") -> None:
        if arr is None or arr.size == 0:
            return
        key = int(np.mean(arr)) if np is not None else hash(arr.tobytes())
        if key in seen:
            return
        seen.add(key)
        variants.append(arr)

    if cv2 is None or np is None:
        _add(np.asarray(cropped.convert("RGB")))
        return variants

    for src_pil in (cropped, img):
        bgr = _pil_to_bgr_uint8(src_pil)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blue = _blue_panel_mask(hsv)
        blue_ratio = float((blue > 0).sum()) / float(blue.size)

        _add(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        if blue_ratio > 0.06:
            v = hsv[:, :, 2].copy()
            v = _shadow_normalize_gray(v)
            v = _clahe_gray(v)
            try:
                adapt = cv2.adaptiveThreshold(
                    v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 7
                )
                adapt = _ensure_dark_text_on_light(adapt)
                _add(cv2.cvtColor(adapt, cv2.COLOR_GRAY2RGB))
            except Exception:
                pass

            white_on_blue = np.full_like(gray, 255)
            bright = cv2.inRange(hsv, (0, 0, 140), (180, 110, 255))
            bright = cv2.bitwise_and(bright, blue)
            white_on_blue[bright > 0] = 0
            _add(cv2.cvtColor(white_on_blue, cv2.COLOR_GRAY2RGB))

        for src in _street_gray_sources(bgr):
            for enhanced in (src, _clahe_gray(src), _shadow_normalize_gray(src)):
                for inverted in (False, True):
                    g = (255 - enhanced) if inverted else enhanced
                    _add(cv2.cvtColor(_binarize_gray(g), cv2.COLOR_GRAY2RGB))
                    if len(variants) >= 10:
                        break
                if len(variants) >= 10:
                    break
            if len(variants) >= 10:
                break
        if len(variants) >= 10:
            break

    return variants[:10] if variants else [np.asarray(cropped.convert("RGB"))]


def _parse_easyocr_hit(
    item: Any, default_conf: float = 0.75
) -> Optional[Tuple[Any, str, float]]:
    """EasyOCR: detail=1 - (bbox, text, conf); paragraph=True - (bbox, text)."""
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    if len(item) >= 3:
        bbox, text, conf = item[0], item[1], item[2]
        return bbox, str(text or ""), float(conf)
    first, second = item[0], item[1]
    if isinstance(first, (list, tuple)) and first and isinstance(first[0], (list, tuple, int, float)):
        return first, str(second or ""), default_conf
    if isinstance(second, (list, tuple)) and second and isinstance(second[0], (list, tuple, int, float)):
        return second, str(first or ""), default_conf
    try:
        return None, str(first or ""), float(second)
    except (TypeError, ValueError):
        return None, str(second or ""), default_conf


def _easyocr_read_lines(reader: Any, arr: "np.ndarray", min_conf: float) -> List[str]:
    """Распознать строки на одном варианте изображения."""
    for paragraph in (False, True):
        try:
            raw = reader.readtext(arr, detail=1, paragraph=paragraph)
        except Exception:
            continue
        items: List[Tuple[float, float, str, float]] = []
        for hit in raw:
            parsed = _parse_easyocr_hit(hit)
            if parsed is None:
                continue
            bbox, text, conf = parsed
            t = (text or "").strip()
            if not t or float(conf) < min_conf:
                continue
            if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                cy = sum(p[1] for p in bbox) / 4.0
                cx = sum(p[0] for p in bbox) / 4.0
            else:
                cy, cx = 0.0, 0.0
            items.append((cy, cx, t, float(conf)))
        if not items:
            continue
        if paragraph:
            return [t for _, _, t, _ in sorted(items, key=lambda x: (x[0], x[1]))]
        grouped = _group_easyocr_items([(cy, t, c) for cy, _cx, t, c in items])
        if grouped:
            return grouped
    return []


def _ocr_street_easyocr(pil: Image.Image) -> str:
    reader = _get_easyocr_reader()
    if reader is None or np is None:
        return ""

    min_conf = _env_int("DIPLOM_OCR_EASYOCR_MIN_CONF", 25) / 100.0
    candidates: List[str] = []

    for arr in _easyocr_input_variants(pil):
        lines = _easyocr_read_lines(reader, arr, min_conf)
        if not lines:
            continue
        joined_nl = "\n".join(_post_process_text(x) for x in lines if x.strip())
        joined_sp = " ".join(_post_process_text(x) for x in lines if x.strip())
        if joined_nl:
            candidates.append(joined_nl)
        if joined_sp and joined_sp != joined_nl.replace("\n", " "):
            candidates.append(joined_sp)

    if not candidates:
        return ""

    merged = _merge_street_line_candidates(candidates)
    if _is_garbage_street_ocr(merged):
        for cand in sorted(candidates, key=_score_text, reverse=True):
            fixed = _post_process_text(cand)
            if fixed and not _is_garbage_street_ocr(fixed):
                merged = fixed
                break
        else:
            return ""
    return _post_process_text(merged)


def _is_garbage_street_ocr(text: str) -> bool:
    """Отсечь «1 1 7 1 74 1» с кирпича и прочий шум без слов."""
    s = _post_process_text(text)
    if not s:
        return True
    parts = s.split()
    if not parts:
        return True
    digit_only = sum(1 for p in parts if p.isdigit())
    long_words = sum(1 for p in parts if re.search(r"[а-яa-z]{4,}", p, re.I))
    cyr_words = sum(1 for p in parts if re.search(r"[а-яё]{3,}", p, re.I))
    # Много одиночных цифр, нет нормальных слов (кирпич / текстура).
    if digit_only >= 3 and long_words == 0:
        return True
    if re.search(r"(?:\b\d\b\s+){3,}", s) and cyr_words == 0:
        return True
    if len(s) <= 2 and not re.search(r"\d{2,}", s):
        return True
    # Частичное слово вроде «Казбекск» — не мусор.
    if long_words >= 1 or cyr_words >= 1:
        return False
    return False


def _crop_sign_panel(pil: Image.Image) -> Image.Image:
    """
    Обрезать кадр до цветной таблички (синяя) или контрастной панели с текстом.
    Убирает кирпич / фон вокруг — главный источник ложных цифр.
    """
    if cv2 is None or np is None:
        return pil

    img = pil_from_rgb(pil)
    w0, h0 = img.size
    if w0 < 40 or h0 < 40:
        return pil

    bgr = _pil_to_bgr_uint8(img)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue_mask = _blue_panel_mask(hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

    best_box: Optional[Tuple[int, int, int, int]] = None
    best_score = -1.0

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        score = _score_panel_contour(cnt, blue_mask, w0, h0)
        if score > best_score:
            best_score = score
            best_box = cv2.boundingRect(cnt)

    # Белая/светлая прямоугольная панель (чёрный текст на белом).
    if best_box is None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel, iterations=2)
        min_area = max(0.02 * w0 * h0, 800.0)
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            ar = bw / max(bh, 1)
            if ar < 1.2 or ar > 8.0:
                continue
            if area > best_score:
                best_score = area
                best_box = (x, y, bw, bh)

    if best_box is None:
        return pil

    x, y, bw, bh = best_box
    # Чуть ужимаем рамку внутрь — меньше кирпича по краям таблички.
    inset_x = max(2, int(bw * 0.04))
    inset_y = max(2, int(bh * 0.03))
    x += inset_x
    y += inset_y
    bw = max(30, bw - 2 * inset_x)
    bh = max(20, bh - 2 * inset_y)
    pad_x = max(4, int(bw * 0.02))
    pad_y = max(4, int(bh * 0.03))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w0, x + bw + pad_x)
    y2 = min(h0, y + bh + pad_y)
    if x2 - x1 < 30 or y2 - y1 < 20:
        return pil
    return img.crop((x1, y1, x2, y2))


def _post_process_text(t: str) -> str:
    s = (t or "").strip()
    if not s:
        return ""
    s = s.replace("„", "").replace("“", "").replace("”", "").replace("«", "").replace("»", "")
    s = s.replace("\\", " ")
    s = s.replace("—", "-")
    s = re.sub(r"\bули\s+ца\b", "улица", s, flags=re.IGNORECASE)
    s = re.sub(r"\bул\s+ица\b", "улица", s, flags=re.IGNORECASE)
    s = re.sub(r"\bказбекс\s*кая\b", "Казбекская", s, flags=re.IGNORECASE)
    s = re.sub(r"\bказбекск(?:ая|ой|ую)?\b", "Казбекская", s, flags=re.IGNORECASE)
    s = re.sub(r"\bул\.?\b", "улица", s, flags=re.IGNORECASE)

    s = " ".join(s.split())
    return s.strip()


def _clahe_gray(gray: "np.ndarray") -> "np.ndarray":
    if cv2 is None:
        return gray
    try:
        return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    except Exception:
        return gray


def _shadow_normalize_gray(gray: "np.ndarray") -> "np.ndarray":
    """Сгладить диагональные тени (деление на размытый фон)."""
    if cv2 is None or np is None:
        return gray
    try:
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
        blur = np.clip(blur.astype(np.float32), 8.0, 255.0)
        norm = (gray.astype(np.float32) / blur) * 140.0
        return np.clip(norm, 0, 255).astype(np.uint8)
    except Exception:
        return gray


def _ensure_dark_text_on_light(bin_img: "np.ndarray") -> "np.ndarray":
    if np is None:
        return bin_img
    dark = int((bin_img < 128).sum())
    light = int((bin_img >= 128).sum())
    if dark > light:
        return cv2.bitwise_not(bin_img) if cv2 is not None else 255 - bin_img
    return bin_img


def _binarize_gray(gray: "np.ndarray") -> "np.ndarray":
    if cv2 is None:
        return gray
    try:
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu = _ensure_dark_text_on_light(otsu)
        ad = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8,
        )
        ad = _ensure_dark_text_on_light(ad)
        # Слегка соединяем разорванные буквы (тень режет штрихи).
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        ad = cv2.morphologyEx(ad, cv2.MORPH_CLOSE, k, iterations=1)
        # Берём вариант с большим «содержательным» счётом — грубо по числу тёмных пикселей в разумных пределах.
        if 0.02 * otsu.size < (otsu < 128).sum() < 0.35 * otsu.size:
            return otsu
        return ad
    except Exception:
        return gray


def _street_gray_sources(bgr: "np.ndarray") -> List["np.ndarray"]:
    """Каналы для белого текста на цветном фоне (синяя табличка и т.п.)."""
    if cv2 is None:
        return []
    out: List["np.ndarray"] = []
    out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    out.append(np.max(bgr, axis=2).astype(np.uint8))
    try:
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2])
        out.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[:, :, 0])
    except Exception:
        pass
    return out


def _street_preprocess_variants(pil: Image.Image) -> List[Image.Image]:
    """
    Несколько бинаризаций: обычный текст, белый на синем, тени, инверсия.
    """
    img = pil_from_rgb(pil)
    img = maybe_upscale(img, min_side=_env_int("DIPLOM_OCR_TEXT_MIN_SIDE", 1000))

    if cv2 is None or np is None:
        return [enhance_for_ocr(img)]

    bgr = _pil_to_bgr_uint8(img)
    variants: List[Image.Image] = []
    seen: set[int] = set()

    def _add(bin_arr: "np.ndarray") -> None:
        key = int(np.mean(bin_arr))
        if key in seen:
            return
        seen.add(key)
        variants.append(Image.fromarray(bin_arr, mode="L"))

    for src in _street_gray_sources(bgr):
        for enhanced in (src, _clahe_gray(src), _shadow_normalize_gray(src)):
            for inverted in (False, True):
                g = (255 - enhanced) if inverted else enhanced
                try:
                    g = cv2.bilateralFilter(g, 5, 30, 30)
                except Exception:
                    pass
                _add(_binarize_gray(g))

    return variants[:10] if variants else [enhance_for_ocr(img)]


def enhance_for_street_sign(pil: Image.Image) -> Image.Image:
    """Лучший одиночный вариант (для совместимости)."""
    variants = _street_preprocess_variants(pil)
    return variants[0]


def _find_text_bands(binary_l: Image.Image, min_band_height: int = 10, gap_merge: int = 8) -> List[Tuple[int, int]]:
    """Найти горизонтальные полосы с текстом (для многострочных табличек)."""
    if cv2 is None or np is None:
        return []
    arr = np.asarray(binary_l)
    if arr.ndim != 2 or arr.shape[0] < min_band_height:
        return []
    # Текст — тёмные пиксели на светлом фоне.
    ink = 255 - arr
    row_sum = ink.sum(axis=1).astype(np.float32)
    peak = float(row_sum.max()) if row_sum.size else 0.0
    if peak <= 0:
        return []
    threshold = max(peak * 0.06, 30.0)
    raw: List[Tuple[int, int]] = []
    in_band = False
    start = 0
    for y, val in enumerate(row_sum):
        if val >= threshold:
            if not in_band:
                start = y
                in_band = True
        elif in_band:
            raw.append((start, y))
            in_band = False
    if in_band:
        raw.append((start, len(row_sum)))

    merged: List[Tuple[int, int]] = []
    for band in raw:
        if merged and band[0] - merged[-1][1] <= gap_merge:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)

    h = arr.shape[0]
    out: List[Tuple[int, int]] = []
    for y0, y1 in merged:
        if y1 - y0 < min_band_height:
            continue
        out.append((max(0, y0 - 3), min(h, y1 + 3)))
    return out


def _ocr_street_band(band_img: Image.Image, lang: str) -> str:
    if pytesseract is None:
        return ""
    extra_cfg = (
        f"-c preserve_interword_spaces=1 "
        f"-c tessedit_char_whitelist={_RU_TEXT_WHITELIST}"
    )
    best = ""
    best_key = (-1.0, -1)
    for psm in (7, 6, 8, 13):
        mean_conf, text = _mean_word_confidence(band_img, lang, psm, extra_cfg=extra_cfg)
        ts = _score_text(text)
        key = (mean_conf, ts)
        if text and key > best_key:
            best_key = key
            best = text
    return _post_process_text(best)


def _ocr_street_tesseract(pil: Image.Image, lang: Optional[str] = None) -> str:
    """Tesseract: построчный OCR табличек (запасной движок)."""
    if pytesseract is None:
        return ""

    lang = lang or _env_street_lang()
    img = pil_from_rgb(pil)
    img = rotate_via_osd(img)
    img = _crop_sign_panel(img)

    candidates: List[str] = []
    extra_cfg = (
        f"-c preserve_interword_spaces=1 "
        f"-c tessedit_char_whitelist={_RU_TEXT_WHITELIST}"
    )

    def _accept_line(line: str) -> bool:
        line = line.strip()
        if len(line) >= 2:
            return True
        return line.isdigit() and len(line) <= 3

    def _ocr_binary_variant(binary: Image.Image) -> None:
        bands = _find_text_bands(binary)
        if len(bands) >= 2:
            lines: List[str] = []
            for y0, y1 in bands:
                crop = binary.crop((0, y0, binary.width, y1))
                line = _ocr_street_band(crop, lang)
                if _accept_line(line):
                    lines.append(line)
            if lines:
                candidates.append("\n".join(lines))
                candidates.append(" ".join(lines))

        whole = deskew_small_angle(binary.convert("RGB"))
        whole_text, _ = _run_ocr_variant(whole, lang, extra_cfg)
        if whole_text:
            candidates.append(whole_text)

    for binary in _street_preprocess_variants(img):
        _ocr_binary_variant(binary)

    # Мягкий вариант без жёсткой бинаризации.
    soft = deskew_small_angle(enhance_for_ocr(maybe_upscale(img)).convert("RGB"))
    soft_text, _ = _run_ocr_variant(soft, lang, extra_cfg="")
    if soft_text:
        candidates.append(soft_text)

    if not candidates:
        return ""

    best = max(candidates, key=_score_text)
    best = _post_process_text(best)
    if _is_garbage_street_ocr(best):
        return ""
    return best


def _ocr_street_sign(pil: Image.Image, lang: Optional[str] = None) -> str:
    """Таблички: EasyOCR (если есть) + Tesseract, выбираем лучший."""
    engine_pref = _env_ocr_engine()
    results: List[Tuple[str, str]] = []

    if engine_pref in ("auto", "easyocr") and easyocr_available():
        ez = _ocr_street_easyocr(pil)
        if ez:
            results.append(("easyocr", ez))

    if engine_pref in ("auto", "tesseract"):
        ts = _ocr_street_tesseract(pil, lang=lang)
        if ts:
            results.append(("tesseract", ts))

    if not results:
        _set_ocr_engine("none")
        return ""

    best_engine, best_text = max(results, key=lambda pair: _score_text(pair[1]))
    _set_ocr_engine(best_engine)
    return best_text


def _run_ocr_variant(
    src_pil: Image.Image, lang: str, extra_cfg: str = ""
) -> Tuple[str, Tuple[float, int]]:
    best_text_local = ""
    best_key_local: Tuple[float, int] = (-1.0, -1)
    for psm in (7, 6, 4, 11, 13):
        mean_conf, text = _mean_word_confidence(src_pil, lang, psm, extra_cfg=extra_cfg)
        ts = _score_text(text)
        key = (mean_conf, ts)
        if key > best_key_local and text:
            best_key_local = key
            best_text_local = text
    if best_text_local and len(best_text_local.strip()) <= 4:
        alt_best = best_text_local
        alt_key = best_key_local
        for psm in (4, 6, 11, 12, 13):
            mean_conf, text = _mean_word_confidence(src_pil, lang, psm, extra_cfg=extra_cfg)
            if not text:
                continue
            ts = _score_text(text)
            if ts >= 40 and (mean_conf + 35.0, ts) > alt_key:
                alt_key = (mean_conf + 35.0, ts)
                alt_best = text
        best_text_local = alt_best
    return best_text_local, best_key_local


def _ocr_plate(pil: Image.Image, lang: str) -> str:
    if pytesseract is None or Output is None:
        return ""
    img = enhance_for_plate(pil)
    # For plates: treat as a single line / single word; narrow whitelist.
    whitelist = _RU_PLATE_ALLOWED
    cfg_base = (
        f"--oem 3 -c user_defined_dpi=300 "
        f"-c tessedit_char_whitelist={whitelist} "
        f"-c preserve_interword_spaces=1"
    )
    best = ""
    best_key = (-1.0, -1)
    for psm in (7, 8, 6):
        cfg = f"{cfg_base} --psm {psm}"
        try:
            txt = (pytesseract.image_to_string(img, lang=lang, config=cfg) or "").strip()
        except Exception:
            txt = ""
        plate = _extract_best_plate(txt)
        if plate:
            key = (2.0 - (0.1 * psm), len(plate))
            if key > best_key:
                best_key = key
                best = plate
    if best:
        return best
    try:
        txt = (pytesseract.image_to_string(img, lang=lang, config=f"{cfg_base} --psm 6") or "").strip()
    except Exception:
        txt = ""
    return _normalize_plate_text(txt)


def run_ocr_on_pil(pil: Image.Image, lang: Optional[str] = None, hint: Optional[str] = None) -> str:
    """
    Full pipeline: EasyOCR для табличек, Tesseract для номеров / fallback.
    """
    lang = lang or _env_lang()
    hint_n = (hint or "").strip().lower()

    if hint_n in ("plate", "license_plate", "car_plate"):
        if pytesseract is None:
            raise RuntimeError("Для номеров нужен pytesseract")
        _set_ocr_engine("tesseract")
        return _ocr_plate(pil, lang=lang)

    if hint_n in ("street", "sign", "text", "highway"):
        if not easyocr_available() and pytesseract is None:
            raise RuntimeError("Установите easyocr или pytesseract")
        return _ocr_street_sign(pil, lang=_env_street_lang())

    if pytesseract is None and not easyocr_available():
        raise RuntimeError("Установите: python -m pip install easyocr")

    if pytesseract is None:
        _set_ocr_engine("easyocr")
        return _ocr_street_easyocr(pil)

    img = pil_from_rgb(pil)
    img = maybe_upscale(img)
    img = rotate_via_osd(img)

    img1 = enhance_for_ocr(img)
    img1 = deskew_small_angle(img1.convert("RGB"))
    best_text, _best_key = _run_ocr_variant(img1, lang, extra_cfg="")

    best_text2 = ""
    if cv2 is not None and np is not None:
        try:
            img2 = enhance_for_street_sign(img)
            img2 = deskew_small_angle(img2.convert("RGB"))
            extra_cfg = (
                f"-c preserve_interword_spaces=1 "
                f"-c tessedit_char_whitelist={_RU_TEXT_WHITELIST}"
            )
            best_text2, _best_key2 = _run_ocr_variant(img2, lang, extra_cfg=extra_cfg)
        except Exception:
            best_text2 = ""

    if _score_text(best_text2) > _score_text(best_text):
        best_text = best_text2

    # Слабый Tesseract - EasyOCR / street pipeline.
    if _score_text(best_text) < 25:
        if easyocr_available():
            ez_try = _ocr_street_easyocr(img)
            if _score_text(ez_try) > _score_text(best_text):
                best_text = ez_try
                _set_ocr_engine("easyocr")
        if _score_text(best_text) < 25:
            try:
                street_try = _ocr_street_sign(img, lang=_env_street_lang())
                if _score_text(street_try) > _score_text(best_text):
                    best_text = street_try
            except Exception:
                pass

    best_text = _post_process_text(best_text)

    if not best_text:
        try:
            best_text = (
                pytesseract.image_to_string(
                    img,
                    lang=lang,
                    config="--oem 3 --psm 6 -c user_defined_dpi=300",
                )
                or ""
            ).strip()
        except Exception:
            best_text = ""

    if _LAST_OCR_ENGINE == "none" and best_text:
        _set_ocr_engine("tesseract")
    return _post_process_text(best_text)


def run_ocr_on_png_bytes(png_bytes: bytes, lang: Optional[str] = None) -> str:
    pil = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    return run_ocr_on_pil(pil, lang=lang)
