"""
Пакетный импорт Geofabrik *.osm.pbf в geo.db (через import_osm_addresses.parse_pbf).

Запуск из папки DIPLOM:
  python import_all_regions.py

Все файлы из ./data/*.osm.pbf:
  python import_all_regions.py

Явный список:
  python import_all_regions.py ^
    "c:\\...\\data\\central-fed-district-260508.osm.pbf" ^
    "c:\\...\\data\\volga-fed-district-260508.osm.pbf"

Лог по умолчанию дописывается в import_all_regions.log рядом со скриптом.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from datetime import datetime, timezone


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    ap = argparse.ArgumentParser(
        description="Импорт всех региональных PBF в geo.db с логом и прогрессом."
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="Явные пути к .osm.pbf; если пусто — сканирование --data-dir",
    )
    ap.add_argument(
        "--data-dir",
        default=os.path.join(here, "data"),
        help="Каталог с PBF, если paths не заданы (по умолчанию: ./data)",
    )
    ap.add_argument(
        "--glob",
        default="*.osm.pbf",
        help="Маска внутри --data-dir (по умолчанию: *.osm.pbf)",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SUBSTR",
        help="Пропускать файлы, в имени которых есть подстрока (можно несколько раз)",
    )
    ap.add_argument(
        "--default-city",
        default=None,
        help="Только для отладки; для РФ не указывать",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить число импортируемых адресов на файл (отладка)",
    )
    ap.add_argument(
        "--log",
        default=None,
        help="Файл лога (по умолчанию: import_all_regions.log рядом со скриптом)",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Остановиться на первой ошибке (по умолчанию: идём дальше)",
    )
    args = ap.parse_args()

    log_path = args.log or os.path.join(here, "import_all_regions.log")

    def log(msg: str) -> None:
        stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    if args.paths:
        files = [os.path.abspath(os.path.expanduser(p)) for p in args.paths]
    else:
        data_dir = os.path.abspath(os.path.expanduser(args.data_dir))
        pattern = os.path.join(data_dir, args.glob)
        files = sorted(glob.glob(pattern))

    for ex in args.exclude:
        ex_l = ex.lower()
        files = [f for f in files if ex_l not in os.path.basename(f).lower()]

    files = [f for f in files if os.path.isfile(f)]

    if not files:
        log("Нет .osm.pbf: задать paths или положить файлы в --data-dir.")
        return 2

    try:
        from import_osm_addresses import parse_pbf
    except ImportError as e:
        log(f"Не удалось импортировать import_osm_addresses: {e!r}")
        log("Запускать из каталога DIPLOM: cd ...\\DIPLOM && python import_all_regions.py")
        return 2

    log(f"Старт: {len(files)} файл(ов) → {os.path.join(here, 'geo.db')}")
    log(f"Лог: {log_path}")

    t_batch = time.perf_counter()
    ok = 0
    failures: list[tuple[str, str]] = []

    for i, pbf in enumerate(files, start=1):
        log(f"--- [{i}/{len(files)}] {pbf} ---")
        t1 = time.perf_counter()
        try:
            c, s, h = parse_pbf(
                pbf,
                default_city=args.default_city,
                limit=args.limit,
            )
            dt = time.perf_counter() - t1
            log(
                f"Готово за {dt / 60.0:.1f} мин (≈ новых сущностей: cities={c}, streets={s}, houses~={h})"
            )
            ok += 1
        except Exception as e:
            log(f"ОШИБКА: {e!r}")
            failures.append((pbf, repr(e)))
            if args.stop_on_error:
                log("Остановка по --stop-on-error.")
                return 1

    total_min = (time.perf_counter() - t_batch) / 60.0
    log(f"=== Итого: успешно {ok}/{len(files)}, время {total_min:.1f} мин ===")
    if failures:
        log(f"Сбои ({len(failures)}):")
        for p, err in failures:
            log(f"  {p}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
