import os
import sqlite3


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(here, "schema_sqlite.sql")
    db_path = os.path.join(here, "geo.db")

    if not os.path.isfile(schema_path):
        raise FileNotFoundError(schema_path)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = f.read()

    con = sqlite3.connect(db_path)
    try:
        con.executescript(schema)
        con.commit()
    finally:
        con.close()

    print(f"OK: created/updated {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

