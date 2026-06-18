"""
Графики и замеры для отчёта.

Запуск из папки DIPLOM:
  python generate_report_charts.py

PNG сохраняются в ../отчет/graphs/
"""

from __future__ import annotations

import csv
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "отчет" / "graphs"


def _import_server():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from server import LocateRequest, locate, open_geo_db, resolve_geo_db_path

    return LocateRequest, locate, open_geo_db, resolve_geo_db_path


def db_counts(con: sqlite3.Connection) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for table in ("cities", "streets", "houses", "intersections"):
        row = con.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()
        counts[table] = int(row[0] if row else 0)
    return counts


def benchmark_queries(locate_fn, LocateRequest) -> Dict[str, List[float]]:
    samples: Dict[str, List[str]] = {
        "city": ["Москва", "СПб", "Новосибирск", "Краснодар", "Екатеринбург"] * 10,
        "street_centroid": [
            "Москва Тверская",
            "Санкт-Петербург Невский",
            "Новосибирск Красный проспект",
            "Краснодар Красная",
            "Екатеринбург Ленина",
        ]
        * 10,
        "intersection": [
            "Степная Тихвинская",
            "Москва Тверская Арбат",
            "Новосибирск Степная Тихвинская",
            "Санкт-Петербург Невский Мойка",
            "Краснодар Красная Седина",
        ]
        * 10,
        "house": [
            "Новосибирск ул. Александра Невского 40",
            "Москва Тверская 1",
            "Санкт-Петербург Невский 28",
            "Краснодар Красная 122",
            "Екатеринбург Ленина 50",
        ]
        * 10,
    }

    timings: Dict[str, List[float]] = {k: [] for k in samples}
    for kind, texts in samples.items():
        for text in texts:
            req = LocateRequest(created_at_ms=0, manual_text=text)
            t0 = time.perf_counter()
            locate_fn(req)
            timings[kind].append(time.perf_counter() - t0)
    return timings


def regression_top1(locate_fn, LocateRequest, csv_path: Path) -> Tuple[int, int]:
    ok = 0
    total = 0
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if (row.get("enabled") or "1").strip().lower() in ("0", "false", "no", "off"):
                continue
            total += 1
            text = row.get("manual_text") or ""
            resp = locate_fn(LocateRequest(created_at_ms=0, manual_text=text))
            best = resp.get("best")
            if best is None:
                continue
            kind_want = (row.get("expect_kind") or "").strip()
            allowed = [k.strip() for k in kind_want.split("|") if k.strip()]
            got = str(best.get("kind", ""))
            if allowed and got not in allowed:
                continue
            name_n = str(best.get("name", "")).lower().replace("ё", "е")
            parts = [p.strip() for p in (row.get("expect_contains_all") or "").split("|") if p.strip()]
            if all(p.lower().replace("ё", "е") in name_n for p in parts):
                ok += 1
    return ok, total


def save_charts(counts: Dict[str, int], timings: Dict[str, List[float]], top1: Tuple[int, int]) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен: pip install matplotlib")
        return []

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    # 1) DB volumes
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["cities", "streets", "houses", "intersections"]
    vals = [counts.get(k, 0) for k in labels]
    ax.bar(labels, vals, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_title("Объём офлайн-базы (SQLite)")
    ax.set_ylabel("Количество записей")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}".replace(",", " "), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p1 = OUT_DIR / "chart_db_volumes.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved.append(p1)

    # 2) Latency medians
    fig, ax = plt.subplots(figsize=(7, 4))
    kinds = list(timings.keys())
    medians = [statistics.median(timings[k]) for k in kinds]
    ax.bar(kinds, medians, color="#72B7B2")
    ax.set_title("Медиана времени ответа locate()")
    ax.set_ylabel("секунды")
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds, rotation=15, ha="right")
    for i, v in enumerate(medians):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p2 = OUT_DIR / "chart_latency.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved.append(p2)

    # 3) Regression top-1
    ok, total = top1
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["top-1 success", "fail"], [ok, max(0, total - ok)], color=["#54A24B", "#E45756"])
    ax.set_title(f"Регрессионные кейсы (n={total})")
    ax.set_ylabel("количество")
    fig.tight_layout()
    p3 = OUT_DIR / "chart_regression_top1.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved.append(p3)

    return saved


def main() -> int:
    LocateRequest, locate_fn, open_geo_db, resolve_geo_db_path = _import_server()

    db_path = resolve_geo_db_path()
    if not db_path or not os.path.isfile(db_path):
        print("geo.db не найден", file=sys.stderr)
        return 2

    con = open_geo_db()
    if con is None:
        print("не удалось открыть geo.db", file=sys.stderr)
        return 2
    try:
        counts = db_counts(con)
    finally:
        con.close()

    print("=== Объёмы БД ===")
    for k, v in counts.items():
        print(f"  {k}: {v:,}".replace(",", " "))

    print("\n=== Замер locate() ===")
    timings = benchmark_queries(locate_fn, LocateRequest)
    for kind, vals in timings.items():
        print(
            f"  {kind}: median={statistics.median(vals):.4f}s "
            f"p95={sorted(vals)[int(0.95 * len(vals)) - 1]:.4f}s "
            f"n={len(vals)}"
        )

    csv_path = HERE / "regression_cases.csv"
    top1 = regression_top1(locate_fn, LocateRequest, csv_path)
    print(f"\n=== Регрессия top-1: {top1[0]}/{top1[1]} ===")

    saved = save_charts(counts, timings, top1)
    if saved:
        print("\nГрафики:")
        for p in saved:
            print(f"  {p}")

    # machine-readable summary for report paste
    summary = OUT_DIR / "metrics_summary.txt"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary, "w", encoding="utf-8") as f:
        f.write("DB counts\n")
        for k, v in counts.items():
            f.write(f"{k}={v}\n")
        f.write("\nLatency median (s)\n")
        for kind, vals in timings.items():
            f.write(f"{kind}={statistics.median(vals):.6f}\n")
        f.write(f"\nregression_top1={top1[0]}/{top1[1]}\n")
    print(f"\nСводка: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
