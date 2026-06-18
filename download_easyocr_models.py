"""
Скачать модели EasyOCR для ru (один раз, ~100–200 МБ).

Запуск из папки DIPLOM:
  python download_easyocr_models.py

На Windows иногда падает SSL у встроенного загрузчика EasyOCR —
этот скрипт качает zip напрямую и распаковывает в ~/.EasyOCR/model.
"""

from __future__ import annotations

import hashlib
import os
import ssl
import sys
import tempfile
import zipfile
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# craft (detector) + cyrillic_g2 (ru + латиница для en в паре с ru)
def _model_specs() -> tuple:
    return (
        {
            "filename": "craft_mlt_25k.pth",
            "urls": (
                "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
                "https://huggingface.co/xiaoyao9184/easyocr/resolve/master/craft_mlt_25k.pth",
            ),
            "md5": "2f8227d2def4037cdb3b34389dcf9ec1",
            "zip": True,
        },
        {
            "filename": "cyrillic_g2.pth",
            "urls": (
                "https://github.com/JaidedAI/EasyOCR/releases/download/v1.6.1/cyrillic_g2.zip",
            ),
            "md5": "19f85f43d9128a89ac21b8d6a06973fe",
            "zip": True,
        },
    )


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_bytes(url: str, timeout: int = 600, retries: int = 4) -> bytes:
    req = Request(url, headers={"User-Agent": "DIPLOM-easyocr-download/1.0"})
    last_err: Optional[BaseException] = None
    ssl_warned = False

    for attempt in range(1, retries + 1):
        for mode in ("default", "certifi", "insecure"):
            try:
                if mode == "default":
                    with urlopen(req, timeout=timeout) as resp:
                        return resp.read()
                if mode == "certifi":
                    import certifi  # type: ignore

                    ctx = ssl.create_default_context(cafile=certifi.where())
                    with urlopen(req, timeout=timeout, context=ctx) as resp:
                        return resp.read()
                if not ssl_warned:
                    print("Предупреждение: SSL verify failed, повтор без проверки сертификата…")
                    ssl_warned = True
                ctx = ssl._create_unverified_context()
                with urlopen(req, timeout=timeout, context=ctx) as resp:
                    return resp.read()
            except Exception as e:
                last_err = e
                continue
        if attempt < retries:
            wait = min(30, 3 * attempt)
            print(f"  повтор {attempt + 1}/{retries} через {wait} с… ({last_err!r})")
            import time

            time.sleep(wait)

    raise RuntimeError(f"Не удалось скачать {url}: {last_err!r}") from last_err


def _extract_pth_from_zip(zip_path: str, dest_dir: str, expected_name: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith(".pth")]
        if not names:
            raise RuntimeError(f"В архиве нет .pth: {zip_path}")
        # Обычно один файл с нужным именем
        chosen = None
        for n in names:
            if os.path.basename(n) == expected_name:
                chosen = n
                break
        if chosen is None:
            chosen = names[0]
        out_path = os.path.join(dest_dir, expected_name)
        with zf.open(chosen) as src, open(out_path, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
        return out_path


def _save_pth(model_dir: str, filename: str, data: bytes, expected_md5: str) -> str:
    out = os.path.join(model_dir, filename)
    os.makedirs(model_dir, exist_ok=True)
    with open(out, "wb") as f:
        f.write(data)
    got = _md5_file(out)
    if got != expected_md5:
        try:
            os.remove(out)
        except OSError:
            pass
        raise RuntimeError(f"MD5 не совпал для {filename}: {got} != {expected_md5}")
    return out


def ensure_model(model_dir: str, spec: dict) -> bool:
    dest = os.path.join(model_dir, spec["filename"])
    if os.path.isfile(dest) and _md5_file(dest) == spec["md5"]:
        print(f"  OK  {spec['filename']}")
        return True

    print(f"  -   {spec['filename']} …")
    last_err: Optional[BaseException] = None
    for url in spec.get("urls") or ():
        try:
            data = _download_bytes(url)
            if spec.get("zip", True) and url.endswith(".zip"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    tmp.write(data)
                    zip_path = tmp.name
                try:
                    out = _extract_pth_from_zip(zip_path, model_dir, spec["filename"])
                finally:
                    try:
                        os.remove(zip_path)
                    except OSError:
                        pass
            elif url.endswith(".pth"):
                out = _save_pth(model_dir, spec["filename"], data, spec["md5"])
            else:
                # zip по содержимому, не по расширению URL
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                    tmp.write(data)
                    zip_path = tmp.name
                try:
                    out = _extract_pth_from_zip(zip_path, model_dir, spec["filename"])
                finally:
                    try:
                        os.remove(zip_path)
                    except OSError:
                        pass
            if _md5_file(out) == spec["md5"]:
                print(f"  OK  {spec['filename']}")
                return True
        except Exception as e:
            last_err = e
            print(f"  !   {url[:60]}… — {e!r}")
            continue
    raise RuntimeError(f"Все зеркала не сработали для {spec['filename']}: {last_err!r}")


def main() -> int:
    try:
        import easyocr
    except ImportError:
        print("Установить пакет: python -m pip install easyocr")
        return 1

    from ocr_pipeline import easyocr_model_dir, easyocr_models_ready

    model_dir = easyocr_model_dir()
    print(f"Папка моделей: {model_dir}")
    if easyocr_models_ready():
        print("Модели уже на месте.")
        return 0

    print("Скачивание (detector + cyrillic). Это может занять несколько минут…")
    try:
        for spec in _model_specs():
            ensure_model(model_dir, spec)
    except Exception as e:
        print(f"Ошибка: {e!r}")
        print(
            "\nЕсли интернет есть, но SSL мешает — этот скрипт уже пробует обход.\n"
            "Иначе скачать вручную zip с GitHub (EasyOCR releases) и положить .pth в:\n"
            f"  {model_dir}"
        )
        return 1

    if easyocr_models_ready():
        print("Готово. Перезапустить GUI.")
        return 0

    print("Файлы скачаны, но проверка не прошла. Повторить команду.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
