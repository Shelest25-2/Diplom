"""
Импорт перекрёстков в geo.db (таблица intersections) из OSM PBF.

Идея: узел, входящий минимум в две именованные линии highway=* с разными name,
      и обе улицы есть в geo.db в одном city_id — записываем точку (lat/lon).

Два прохода по файлу (узлы считаются в памяти — для всей РФ нужен некторый запас RAM).

  python import_osm_intersections.py path\\to\\region.osm.pbf
  python import_osm_intersections.py data\\siberian-fed-district.osm.pbf --max-inserts 20000

Перед запуском уже должен быть прогнан import_osm_addresses (улицы/города из addr:*).

Зависимости: pip install osmium
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
import time
from typing import Dict, FrozenSet, Optional, Set, Tuple

# Те же нормализации, что и при импорте домов.
from import_osm_addresses import normalize_street

HERE = os.path.dirname(os.path.abspath(__file__))

# Узкие классы дорог — меньше шумных «пересечений» и объём счётчика узлов.
HIGHWAY_OK: FrozenSet[str] = frozenset(
    {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "living_street",
        "road",
    }
)


def _load_major_city_names() -> Set[str]:
    names: Set[str] = set()
    path = os.path.join(HERE, "ru_cities.csv")
    if not os.path.isfile(path):
        return names
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n = (row.get("name") or "").strip()
            if n:
                names.add(n)
    return names


def _is_named_highway(way) -> bool:
    t = way.tags
    if t.get("highway") not in HIGHWAY_OK:
        return False
    name = (t.get("name") or "").strip()
    if len(name) < 3 or len(name) > 120:
        return False
    return True


def _pass1_count_nodes(pbf_path: str) -> Dict[int, int]:
    import osmium  # type: ignore

    counts: Dict[int, int] = {}

    class H(osmium.SimpleHandler):  # type: ignore
        def way(self, w) -> None:
            if not _is_named_highway(w):
                return
            seen: Set[int] = set()
            for n in w.nodes:
                rid = int(n.ref)
                if rid in seen:
                    continue
                seen.add(rid)
                counts[rid] = counts.get(rid, 0) + 1

    h = H()
    t0 = time.perf_counter()
    h.apply_file(pbf_path, locations=False)
    dt = time.perf_counter() - t0
    print(f"Pass 1: узлов с учётом дорог: {len(counts):,} за {dt:.1f} с")
    return counts


def _pass2_fill_temp(
    pbf_path: str,
    counts: Dict[int, int],
    tmp_path: str,
) -> sqlite3.Connection:
    import osmium  # type: ignore

    if os.path.isfile(tmp_path):
        os.remove(tmp_path)
    tmp = sqlite3.connect(tmp_path)
    tmp.execute(
        """
        CREATE TABLE nw (
          node_id INTEGER NOT NULL,
          name_norm TEXT NOT NULL,
          UNIQUE(node_id, name_norm)
        );
        """
    )
    tmp.execute(
        """
        CREATE TABLE node_pos (
          node_id INTEGER PRIMARY KEY,
          lat REAL NOT NULL,
          lon REAL NOT NULL
        );
        """
    )
    tmp.execute("CREATE INDEX idx_nw_node ON nw(node_id);")
    tmp.commit()

    batch_nw: list[Tuple[int, str]] = []
    batch_pos: list[Tuple[int, float, float]] = []
    BATCH = 50_000

    def flush() -> None:
        nonlocal batch_nw, batch_pos
        if batch_nw:
            tmp.executemany(
                "INSERT OR IGNORE INTO nw (node_id, name_norm) VALUES (?,?)",
                batch_nw,
            )
            batch_nw = []
        if batch_pos:
            tmp.executemany(
                "INSERT OR REPLACE INTO node_pos (node_id, lat, lon) VALUES (?,?,?)",
                batch_pos,
            )
            batch_pos = []
        tmp.commit()

    class H(osmium.SimpleHandler):  # type: ignore
        def way(self, w) -> None:
            if not _is_named_highway(w):
                return
            name_norm = normalize_street(w.tags["name"])
            if not name_norm or len(name_norm) < 2:
                return
            seen: Set[int] = set()
            for n in w.nodes:
                rid = int(n.ref)
                if rid in seen:
                    continue
                seen.add(rid)
                if counts.get(rid, 0) < 2:
                    continue
                if not n.location or not n.location.valid():
                    continue
                lat = float(n.location.lat)
                lon = float(n.location.lon)
                batch_nw.append((rid, name_norm))
                batch_pos.append((rid, lat, lon))
                if len(batch_nw) >= BATCH:
                    flush()

    h = H()
    t0 = time.perf_counter()
    h.apply_file(pbf_path, locations=True)
    flush()
    dt = time.perf_counter() - t0
    n_nw = int(tmp.execute("SELECT COUNT(*) FROM nw").fetchone()[0])
    print(f"Pass 2: записей nw={n_nw:,} за {dt:.1f} с")
    return tmp


def _pick_city_for_pair(
    geo: sqlite3.Connection,
    a_norm: str,
    b_norm: str,
    major_names: Set[str],
) -> Optional[int]:
    rows = geo.execute(
        """
        SELECT s1.city_id,
               c.name AS city_name,
               (SELECT COUNT(*) FROM houses h WHERE h.city_id = s1.city_id) AS nh
        FROM streets s1
        JOIN streets s2
          ON s1.city_id = s2.city_id AND s1.street_id < s2.street_id
        JOIN cities c ON c.city_id = s1.city_id
        WHERE s1.name_norm = ? AND s2.name_norm = ?
        ORDER BY nh DESC
        """,
        (a_norm, b_norm),
    ).fetchall()

    if not rows:
        return None
    if len(rows) == 1:
        return int(rows[0]["city_id"])

    best_h = int(rows[0]["nh"])
    tied = [r for r in rows if int(r["nh"]) == best_h]
    if len(tied) == 1:
        return int(tied[0]["city_id"])

    for r in tied:
        if str(r["city_name"]) in major_names:
            return int(r["city_id"])
    return int(tied[0]["city_id"])


def _street_ids(geo: sqlite3.Connection, city_id: int, name_norm: str) -> Optional[int]:
    row = geo.execute(
        "SELECT street_id FROM streets WHERE city_id = ? AND name_norm = ? LIMIT 1",
        (city_id, name_norm),
    ).fetchone()
    return int(row["street_id"]) if row else None


def import_intersections(
    pbf_path: str,
    geo_path: Optional[str] = None,
    tmp_path: Optional[str] = None,
    max_inserts: Optional[int] = None,
    keep_temp: bool = False,
) -> int:
    geo_path = geo_path or os.path.join(HERE, "geo.db")
    if not os.path.isfile(geo_path):
        raise FileNotFoundError(geo_path)
    tmp_path = tmp_path or os.path.join(HERE, "_intersections_build.sqlite")
    if not os.path.isfile(pbf_path):
        raise FileNotFoundError(pbf_path)

    major = _load_major_city_names()
    counts = _pass1_count_nodes(pbf_path)
    tmp = _pass2_fill_temp(pbf_path, counts, tmp_path)
    counts.clear()

    geo = sqlite3.connect(geo_path)
    geo.row_factory = sqlite3.Row
    geo.execute("PRAGMA journal_mode=WAL;")
    geo.execute("PRAGMA synchronous=NORMAL;")

    # Caches: drastically reduce per-node SQLite lookups.
    pair_city_cache: Dict[Tuple[str, str], Optional[int]] = {}
    street_id_cache: Dict[Tuple[int, str], Optional[int]] = {}

    inserted = 0
    skipped_no_city = 0
    skipped_exists = 0

    cur = tmp.execute(
        """
        SELECT node_id
        FROM nw
        GROUP BY node_id
        HAVING COUNT(DISTINCT name_norm) >= 2
        """
    )
    candidates = [int(r[0]) for r in cur.fetchall()]
    print(f"Кандидатов узлов (≥2 разных name на узле): {len(candidates):,}")

    t0 = time.perf_counter()
    for i, nid in enumerate(candidates):
        if max_inserts is not None and inserted >= max_inserts:
            break
        if i and i % 10_000 == 0:
            dt = time.perf_counter() - t0
            rate = i / dt if dt > 0 else 0.0
            eta_s = (len(candidates) - i) / rate if rate > 0 else 0.0
            print(
                f"  … узлов {i:,}/{len(candidates):,}, вставок {inserted:,}, "
                f"{rate:.1f} узл/с, ETA ~{eta_s/60.0:.1f} мин"
            )

        names = [
            str(r[0])
            for r in tmp.execute(
                "SELECT DISTINCT name_norm FROM nw WHERE node_id = ? ORDER BY name_norm",
                (nid,),
            ).fetchall()
        ]
        if len(names) < 2:
            continue
        pos = tmp.execute(
            "SELECT lat, lon FROM node_pos WHERE node_id = ?",
            (nid,),
        ).fetchone()
        if not pos:
            continue
        lat, lon = float(pos[0]), float(pos[1])

        pairs: Set[Tuple[str, str]] = set()
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                n1, n2 = names[a], names[b]
                if n1 == n2:
                    continue
                pairs.add((n1, n2) if n1 < n2 else (n2, n1))

        for a_norm, b_norm in pairs:
            if max_inserts is not None and inserted >= max_inserts:
                break
            key = (a_norm, b_norm) if a_norm < b_norm else (b_norm, a_norm)
            if key in pair_city_cache:
                cid = pair_city_cache[key]
            else:
                cid = _pick_city_for_pair(geo, key[0], key[1], major)
                pair_city_cache[key] = cid
            if cid is None:
                skipped_no_city += 1
                continue

            k_a = (cid, a_norm)
            if k_a in street_id_cache:
                sid_a = street_id_cache[k_a]
            else:
                sid_a = _street_ids(geo, cid, a_norm)
                street_id_cache[k_a] = sid_a

            k_b = (cid, b_norm)
            if k_b in street_id_cache:
                sid_b = street_id_cache[k_b]
            else:
                sid_b = _street_ids(geo, cid, b_norm)
                street_id_cache[k_b] = sid_b

            if sid_a is None or sid_b is None or sid_a == sid_b:
                skipped_no_city += 1
                continue
            s1, s2 = (sid_a, sid_b) if sid_a < sid_b else (sid_b, sid_a)

            ex = geo.execute(
                """
                SELECT 1 FROM intersections
                WHERE city_id = ? AND street_a_id = ? AND street_b_id = ?
                LIMIT 1
                """,
                (cid, s1, s2),
            ).fetchone()
            if ex:
                skipped_exists += 1
                continue

            geo.execute(
                """
                INSERT INTO intersections (city_id, street_a_id, street_b_id, lat, lon)
                VALUES (?, ?, ?, ?, ?)
                """,
                (cid, s1, s2, lat, lon),
            )
            inserted += 1
            if inserted % 2000 == 0:
                geo.commit()

    geo.commit()
    geo.close()
    tmp.close()
    if not keep_temp:
        try:
            os.remove(tmp_path)
        except OSError:
            print(f"(не удалось удалить временный файл: {tmp_path})")
    else:
        print(f"Временный файл: {tmp_path}")

    dt = time.perf_counter() - t0
    print(
        f"Готово за {dt:.1f} с: вставлено пересечений {inserted:,}, "
        f"пропуск нет пары в БД {skipped_no_city:,}, уже было {skipped_exists:,}"
    )
    return inserted


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Импорт intersections из OSM PBF в geo.db")
    ap.add_argument("pbf", help="region.osm.pbf")
    ap.add_argument("--geo-db", default=None, help="Путь к geo.db")
    ap.add_argument(
        "--max-inserts",
        type=int,
        default=None,
        help="Остановиться после N вставок (отладка / быстрый прогон)",
    )
    ap.add_argument(
        "--keep-temp",
        action="store_true",
        help="Не удалять временный SQLite после импорта",
    )
    args = ap.parse_args(argv)

    tmp = os.path.join(HERE, "_intersections_build.sqlite")
    import_intersections(
        args.pbf,
        geo_path=args.geo_db,
        tmp_path=tmp,
        max_inserts=args.max_inserts,
        keep_temp=args.keep_temp,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
