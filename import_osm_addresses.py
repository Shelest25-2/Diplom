import os
import re
import sqlite3
import sys
from dataclasses import dataclass
from typing import Optional


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)
REPEAT_RE = re.compile(r"(.)\1{2,}", re.UNICODE)
STREET_TYPE_RE = re.compile(
    r"\b(ул|улица|пр-?т|проспект|пер|переулок|шоссе|бул|бульвар|наб|набережная|пл|площадь)\b\.?",
    re.IGNORECASE | re.UNICODE,
)


def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("ё", "е")
    s = re.sub(r"\s+", " ", s)
    s = REPEAT_RE.sub(r"\1\1", s)
    return s.strip()


def normalize_street(s: str) -> str:
    s = normalize(s)
    s = STREET_TYPE_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_house(s: str) -> str:
    return normalize(s).replace(" ", "")


@dataclass
class AddressRow:
    city: str
    street: str
    house: str
    lat: float
    lon: float


def upsert_city(con: sqlite3.Connection, name: str, lat: float, lon: float) -> int:
    name_norm = normalize(name)
    row = con.execute("SELECT city_id FROM cities WHERE name_norm = ?", (name_norm,)).fetchone()
    if row:
        return int(row[0])
    cur = con.execute(
        "INSERT INTO cities(name, name_norm, lat, lon) VALUES(?,?,?,?)",
        (name.strip(), name_norm, float(lat), float(lon)),
    )
    return int(cur.lastrowid)


def upsert_street(con: sqlite3.Connection, city_id: int, name: str) -> int:
    name_norm = normalize_street(name)
    row = con.execute("SELECT street_id FROM streets WHERE city_id = ? AND name_norm = ?", (city_id, name_norm)).fetchone()
    if row:
        return int(row[0])
    cur = con.execute(
        "INSERT INTO streets(city_id, name, name_norm) VALUES(?,?,?)",
        (city_id, name.strip(), name_norm),
    )
    return int(cur.lastrowid)


def insert_house(con: sqlite3.Connection, city_id: int, street_id: int, house: str, lat: float, lon: float) -> None:
    house_norm = normalize_house(house)
    # Avoid duplicates
    row = con.execute(
        "SELECT 1 FROM houses WHERE city_id = ? AND street_id = ? AND house_norm = ? LIMIT 1",
        (city_id, street_id, house_norm),
    ).fetchone()
    if row:
        return
    con.execute(
        "INSERT INTO houses(city_id, street_id, house, house_norm, lat, lon) VALUES(?,?,?,?,?,?)",
        (city_id, street_id, house.strip(), house_norm, float(lat), float(lon)),
    )


class StopImport(Exception):
    pass


def parse_pbf(pbf_path: str, default_city: Optional[str], limit: Optional[int]) -> tuple[int, int, int]:
    try:
        import osmium  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Не найден Python пакет `osmium`. Установить:\n\n  python -m pip install osmium\n\n"
            f"Ошибка импорта: {e!r}"
        )

    here = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(here, "geo.db")
    if not os.path.isfile(db_path):
        raise FileNotFoundError("geo.db not found. Run init_db.py first.")

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")

    counts = {"cities": 0, "streets": 0, "houses": 0}
    seen_city_ids: set[int] = set()
    seen_street_ids: set[int] = set()

    class Handler(osmium.SimpleHandler):  # type: ignore
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def _process_addr(self, tags, lat: float, lon: float) -> None:  # type: ignore
            hn = tags.get("addr:housenumber")
            st = tags.get("addr:street")
            if not hn or not st:
                return

            city = tags.get("addr:city") or tags.get("addr:place") or default_city
            if not city:
                return

            cid = upsert_city(con, city, lat, lon)
            sid = upsert_street(con, cid, st)
            insert_house(con, cid, sid, hn, lat, lon)

            if cid not in seen_city_ids:
                seen_city_ids.add(cid)
                counts["cities"] += 1
            if sid not in seen_street_ids:
                seen_street_ids.add(sid)
                counts["streets"] += 1
            counts["houses"] += 1

            self.n += 1
            if self.n % 2000 == 0:
                con.commit()
                print(f"Imported addresses: {self.n:,}")
            if limit and self.n >= limit:
                raise StopImport()

        def node(self, n) -> None:  # type: ignore
            if not n.location or not n.location.valid():
                return

            try:
                lat = float(n.location.lat)
                lon = float(n.location.lon)
            except Exception:
                return

            self._process_addr(n.tags, lat, lon)

        def way(self, w) -> None:  # type: ignore
            # Many addresses in OSM are attached to building ways.
            tags = w.tags
            if not tags.get("addr:housenumber") or not tags.get("addr:street"):
                return

            coords: list[tuple[float, float]] = []
            try:
                for nd in w.nodes:
                    if not nd.location or not nd.location.valid():
                        continue
                    coords.append((float(nd.location.lat), float(nd.location.lon)))
            except Exception:
                return
            if not coords:
                return

            # Centroid approximation by average coordinates.
            lat = sum(c[0] for c in coords) / len(coords)
            lon = sum(c[1] for c in coords) / len(coords)
            self._process_addr(tags, lat, lon)

    h = Handler()
    try:
        try:
            h.apply_file(pbf_path, locations=True)
        except StopImport:
            pass
    finally:
        con.commit()
        con.close()

    # counts['houses'] is inserted attempts, not unique rows, but OK for progress.
    return counts["cities"], counts["streets"], counts["houses"]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python import_osm_addresses.py <region.osm.pbf> [--default-city \"Новосибирск\"] [--limit N]")
        return 2

    pbf = argv[1]
    if not os.path.isfile(pbf):
        print("File not found:", pbf)
        return 2

    default_city: Optional[str] = None
    limit: Optional[int] = None
    i = 2
    while i < len(argv):
        if argv[i] == "--default-city" and i + 1 < len(argv):
            default_city = argv[i + 1]
            i += 2
            continue
        if argv[i] == "--limit" and i + 1 < len(argv):
            try:
                limit = int(argv[i + 1])
            except Exception:
                limit = None
            i += 2
            continue
        i += 1

    c, s, h = parse_pbf(pbf, default_city=default_city, limit=limit)
    print(f"Готово. Новых сущностей (approx): cities={c}, streets={s}, houses_rows_inserted~={h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

