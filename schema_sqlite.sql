PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA temp_store = MEMORY;

-- Core entities (offline geocoder-like DB)

CREATE TABLE IF NOT EXISTS cities (
  city_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  name_norm TEXT NOT NULL,
  lat REAL NOT NULL,
  lon REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cities_name_norm ON cities(name_norm);

CREATE TABLE IF NOT EXISTS city_aliases (
  city_id INTEGER NOT NULL REFERENCES cities(city_id) ON DELETE CASCADE,
  alias TEXT NOT NULL,
  alias_norm TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_city_aliases_alias_norm ON city_aliases(alias_norm);

CREATE TABLE IF NOT EXISTS streets (
  street_id INTEGER PRIMARY KEY,
  city_id INTEGER NOT NULL REFERENCES cities(city_id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  name_norm TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_streets_city_name_norm ON streets(city_id, name_norm);

CREATE TABLE IF NOT EXISTS street_aliases (
  street_id INTEGER NOT NULL REFERENCES streets(street_id) ON DELETE CASCADE,
  alias TEXT NOT NULL,
  alias_norm TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_street_aliases_alias_norm ON street_aliases(alias_norm);

CREATE TABLE IF NOT EXISTS houses (
  house_id INTEGER PRIMARY KEY,
  city_id INTEGER NOT NULL REFERENCES cities(city_id) ON DELETE CASCADE,
  street_id INTEGER NOT NULL REFERENCES streets(street_id) ON DELETE CASCADE,
  house TEXT NOT NULL,
  house_norm TEXT NOT NULL,
  lat REAL NOT NULL,
  lon REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_houses_city_street_house_norm ON houses(city_id, street_id, house_norm);

CREATE TABLE IF NOT EXISTS intersections (
  inter_id INTEGER PRIMARY KEY,
  city_id INTEGER NOT NULL REFERENCES cities(city_id) ON DELETE CASCADE,
  street_a_id INTEGER NOT NULL REFERENCES streets(street_id) ON DELETE CASCADE,
  street_b_id INTEGER NOT NULL REFERENCES streets(street_id) ON DELETE CASCADE,
  lat REAL NOT NULL,
  lon REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_intersections_city_pair ON intersections(city_id, street_a_id, street_b_id);

CREATE VIRTUAL TABLE IF NOT EXISTS streets_fts USING fts5(
  street_id UNINDEXED,
  city_id UNINDEXED,
  name,
  aliases,
  tokenize = 'unicode61'
);

