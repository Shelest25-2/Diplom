"""
Прогон фиксированных строк через locate() — регрессия после правок server/GUI/OCR.

Запуск (из папки DIPLOM, нужен geo.db рядом или DIPLOM_GEO_DB):

  python run_regression_tests.py

Свой файл кейсов:

  python run_regression_tests.py --csv my_cases.csv

Формат regression_cases.csv (UTF-8):
  - id: короткое имя
  - enabled: 1 / 0
  - expect_kind: ожидаемый best.kind; можно несколько через | (любой подходит), например intersection|intersection_approx
  - expect_contains_all: подстроки через | — все должны встречаться в best.name (без учёта регистра, ё→е)
  - expect_not_contains: подстроки через | — ни одна не должна быть в best.name
  - manual_text: вход (допускаются переносы строк внутри кавычек в CSV)
  - note: комментарий, не используется скриптом
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _split_pipe(s: str) -> List[str]:
    return [p.strip() for p in (s or "").split("|") if p.strip()]


def _check_row(
    row: Dict[str, str],
    resp: Dict[str, Any],
    normalize_fn,
) -> Tuple[bool, str]:
    best = resp.get("best")
    if best is None:
        return False, "нет best (пустой topk / неоднозначно)"

    kind_want = (row.get("expect_kind") or "").strip()
    if kind_want:
        allowed = [k.strip() for k in kind_want.split("|") if k.strip()]
        got = str(best.get("kind", ""))
        if allowed and got not in allowed:
            return False, f"kind: ожидалось одно из {allowed!r}, получено {got!r}"

    name_n = normalize_fn(str(best.get("name", "")))

    for part in _split_pipe(row.get("expect_contains_all", "")):
        pn = normalize_fn(part)
        if pn and pn not in name_n:
            return False, f"в best.name нет подстроки {part!r} (норм: {pn!r})"

    for part in _split_pipe(row.get("expect_not_contains", "")):
        pn = normalize_fn(part)
        if pn and pn in name_n:
            return False, f"в best.name не должно быть {part!r}"

    return True, ""


def main() -> int:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    ap = argparse.ArgumentParser(description="Регрессионные тесты locate() по CSV.")
    ap.add_argument(
        "--csv",
        default=str(here / "regression_cases.csv"),
        help="Путь к CSV с кейсами",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"Файл не найден: {csv_path}", file=sys.stderr)
        return 2

    os.environ.setdefault("PYTHONUTF8", "1")

    from server import LocateRequest, locate, normalize, resolve_geo_db_path

    db_path = resolve_geo_db_path()
    if not db_path or not os.path.isfile(db_path):
        print(
            "geo.db не найден. Задай DIPLOM_GEO_DB или положи geo.db рядом со server.py.",
            file=sys.stderr,
        )
        return 2

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not reader.fieldnames or "manual_text" not in reader.fieldnames:
        print("CSV: нужна колонка manual_text.", file=sys.stderr)
        return 2

    fails: List[str] = []
    skipped = 0
    ran = 0

    for row in rows:
        rid = (row.get("id") or f"row_{ran}").strip()
        en = (row.get("enabled") or "1").strip().lower()
        if en in ("0", "false", "no", "off"):
            skipped += 1
            continue

        text = row.get("manual_text") or ""
        req = LocateRequest(created_at_ms=0, manual_text=text)
        try:
            resp = locate(req)
        except Exception as e:
            fails.append(f"{rid}: исключение {e!r}")
            ran += 1
            continue

        ok, err = _check_row(row, resp, normalize)
        ran += 1
        if not ok:
            fails.append(f"{rid}: {err} | best={resp.get('best')!r}")

    print(f"geo.db: {db_path}")
    print(f"Кейсов прогнано: {ran}, пропущено (enabled=0): {skipped}, провалов: {len(fails)}")
    for line in fails:
        print(" FAIL:", line)

    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
