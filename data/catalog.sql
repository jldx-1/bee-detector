PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sources (
    name TEXT PRIMARY KEY,
    root_path TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    registered_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- A group's split is stored exactly once, independently of physical folders.
CREATE TABLE IF NOT EXISTS groups (
    name TEXT PRIMARY KEY,
    split TEXT NOT NULL CHECK (split IN ('train', 'val', 'test'))
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY CHECK (id BETWEEN 0 AND 255),
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    color TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL REFERENCES sources(name),
    group_name TEXT NOT NULL REFERENCES groups(name),
    image_path TEXT NOT NULL,
    mask_path TEXT NOT NULL,
    image_sha256 TEXT NOT NULL UNIQUE,
    mask_sha256 TEXT NOT NULL,
    width INTEGER NOT NULL CHECK (width > 0),
    height INTEGER NOT NULL CHECK (height > 0),
    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source, image_path),
    UNIQUE (source, mask_path)
);

CREATE INDEX IF NOT EXISTS samples_group_idx ON samples(group_name);

CREATE TABLE IF NOT EXISTS sample_labels (
    sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
    label_id INTEGER NOT NULL REFERENCES labels(id),
    pixels INTEGER NOT NULL CHECK (pixels > 0),
    PRIMARY KEY (sample_id, label_id)
);

CREATE VIEW IF NOT EXISTS dataset AS
SELECT s.*, g.split, d.root_path, d.collected_at, d.notes AS source_notes
FROM samples s JOIN groups g ON g.name = s.group_name JOIN sources d ON d.name = s.source;

CREATE VIEW IF NOT EXISTS source_summary AS
SELECT d.*, COUNT(s.id) AS images,
    MIN(s.width) AS min_width, MAX(s.width) AS max_width,
    MIN(s.height) AS min_height, MAX(s.height) AS max_height,
    COUNT(CASE WHEN s.split = 'train' THEN 1 END) AS train,
    COUNT(CASE WHEN s.split = 'val' THEN 1 END) AS val,
    COUNT(CASE WHEN s.split = 'test' THEN 1 END) AS test
FROM sources d LEFT JOIN dataset s ON s.source = d.name GROUP BY d.name;

INSERT OR IGNORE INTO labels VALUES
    (0, 'background', 'background', '#E8E8E4'),
    (1, 'worker', 'bee', '#E6A817'),
    (2, 'drone', 'bee', '#468C86'),
    (3, 'queen', 'bee', '#9B6AC6'),
    (4, 'pollen', 'resource', '#F2CD49'),
    (5, 'honey', 'resource', '#CD7926'),
    (6, 'capped_brood', 'brood', '#BAA183'),
    (7, 'uncapped_brood', 'brood', '#E6BDB1'),
    (8, 'varroa_mite', 'pest', '#AD4F42'),
    (9, 'small_hive_beetle', 'pest', '#5D6470'),
    (10, 'wax_moth_larva', 'pest', '#98AA73'),
    (11, 'deformed_wing', 'disease_sign', '#D97084'),
    (12, 'chalkbrood_suspected', 'disease_sign', '#71A4BC'),
    (13, 'foulbrood_suspected', 'disease_sign', '#A87D5D'),
    (255, 'ignore', 'ignore', '#222222');
