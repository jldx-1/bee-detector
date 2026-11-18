use anyhow::{Context, Result, ensure};
use chrono::NaiveDate;
use rusqlite::{Connection, OpenFlags, OptionalExtension, hooks, params, types::ValueRef};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    time::Duration,
};

const SCHEMA: &str = include_str!("../data/catalog.sql");
pub const SPLITS: [&str; 3] = ["train", "val", "test"];
pub const PAGE_SIZE: usize = 24;

pub struct Catalog {
    pub path: PathBuf,
    conn: Connection,
}

#[derive(Clone, Default)]
pub struct Filter {
    pub split: String,
    pub source: String,
    pub search: String,
    pub label: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct Sample {
    pub id: i64,
    pub source: String,
    pub group: String,
    pub split: String,
    pub root: PathBuf,
    pub image: String,
    pub mask: String,
    pub collected: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug)]
pub struct Source {
    pub name: String,
    pub root: String,
    pub collected: String,
    pub notes: String,
    pub images: usize,
    pub resolution: String,
    pub splits: [usize; 3],
}

#[derive(Clone)]
pub struct Label {
    pub id: i64,
    pub name: String,
    pub color: [u8; 3],
    pub images: usize,
}

#[derive(Default)]
pub struct Summary {
    pub splits: [usize; 3],
    pub groups: usize,
    pub sources: Vec<Source>,
    pub labels: Vec<Label>,
}

pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub truncated: bool,
}

pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub ids: Vec<u8>,
}

struct Inspected {
    width: u32,
    height: u32,
    image_hash: String,
    mask_hash: String,
    counts: BTreeMap<u8, u64>,
}

impl Catalog {
    pub fn open(path: &Path) -> Result<Self> {
        let absolute = if path.is_absolute() {
            path.to_owned()
        } else {
            std::env::current_dir()?.join(path)
        };
        fs::create_dir_all(
            absolute
                .parent()
                .context("Catalog needs a parent directory")?,
        )?;
        let conn = Connection::open(&absolute)?;
        conn.busy_timeout(Duration::from_secs(5))?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")?;
        let version: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        ensure!(
            version == 0 || version == 2,
            "Unsupported catalog version {version}"
        );
        // Reject incompatible catalogs instead of changing their existing tables.
        let old: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE name='settings')",
            [],
            |r| r.get(0),
        )?;
        ensure!(
            !old,
            "Unsupported catalog schema; choose a different --catalog path"
        );
        if version == 0 {
            conn.execute_batch(SCHEMA)?;
            conn.pragma_update(None, "user_version", 2)?;
        }
        Ok(Self {
            path: absolute.canonicalize()?,
            conn,
        })
    }

    pub fn root(&self, source: &str) -> Result<PathBuf> {
        let path: String = self
            .conn
            .query_row(
                "SELECT root_path FROM sources WHERE name=?1",
                [source],
                |r| r.get(0),
            )
            .with_context(|| format!("Unknown source: {source}"))?;
        Ok(self.path.parent().unwrap().join(path))
    }

    fn stored_root(&self, path: &Path) -> Result<String> {
        let root = path
            .canonicalize()
            .with_context(|| format!("Source folder not found: {}", path.display()))?;
        ensure!(root.is_dir(), "Source must be a directory");
        let relative = pathdiff::diff_paths(&root, self.path.parent().unwrap()).unwrap_or(root);
        Ok(relative.to_str().context("Use UTF-8 paths")?.to_owned())
    }

    pub fn add_source(&self, name: &str, path: &Path, collected: &str, notes: &str) -> Result<()> {
        validate_source(name, collected)?;
        let root = self.stored_root(path)?;
        self.conn.execute(
            "INSERT INTO sources(name,root_path,collected_at,notes) VALUES (?1,?2,?3,?4)",
            params![name, root, collected, notes],
        )?;
        Ok(())
    }

    pub fn register(
        &mut self,
        source: &str,
        image: &Path,
        mask: &Path,
        split: &str,
        group: &str,
    ) -> Result<i64> {
        let root = self.root(source)?.canonicalize()?;
        let tx = self.conn.transaction()?;
        let id = register_pair(&tx, &root, source, image, mask, split, group)?;
        tx.commit()?;
        Ok(id)
    }

    pub fn import(&mut self, source: &str, split: &str, group: &str) -> Result<usize> {
        let root = self.root(source)?.canonicalize()?;
        let tx = self.conn.transaction()?;
        let mut count = 0;
        for entry in walkdir::WalkDir::new(&root).sort_by_file_name() {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            let image = entry.path();
            let name = image
                .file_name()
                .and_then(|n| n.to_str())
                .context("Use UTF-8 names")?;
            if name.ends_with(".mask.png") {
                continue;
            }
            let extension = image
                .extension()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !matches!(
                extension.as_str(),
                "png" | "jpg" | "jpeg" | "webp" | "tif" | "tiff"
            ) {
                continue;
            }
            let stem = image
                .file_stem()
                .unwrap()
                .to_str()
                .context("Use UTF-8 names")?;
            let mask = image.with_file_name(format!("{stem}.mask.png"));
            ensure!(
                mask.is_file(),
                "Missing mask: {}. Use register for a different file layout.",
                mask.display()
            );
            register_pair(&tx, &root, source, image, &mask, split, group)?;
            count += 1;
        }
        tx.commit()?;
        Ok(count)
    }

    pub fn move_group(&self, group: &str, split: &str) -> Result<usize> {
        ensure!(SPLITS.contains(&split), "Invalid split");
        let count = self.conn.query_row(
            "SELECT COUNT(*) FROM samples WHERE group_name=?1",
            [group],
            |r| r.get(0),
        )?;
        ensure!(
            self.conn.execute(
                "UPDATE groups SET split=?1 WHERE name=?2",
                params![split, group]
            )? == 1,
            "Unknown group"
        );
        Ok(count)
    }

    pub fn relocate(&self, source: &str, path: &Path) -> Result<()> {
        self.root(source)?;
        let root = self.stored_root(path)?;
        let literal: String = self
            .conn
            .query_row("SELECT quote(?1)", [root], |r| r.get(0))?;
        let source: String = self
            .conn
            .query_row("SELECT quote(?1)", [source], |r| r.get(0))?;
        self.apply_sql(&format!(
            "UPDATE sources SET root_path={literal} WHERE name={source}"
        ))
    }

    /// Move a whole source on the same filesystem, preserving SQL on handled failures.
    pub fn move_source(&self, source: &str, destination: &Path) -> Result<()> {
        let old = self.root(source)?.canonicalize()?;
        ensure!(
            !self.path.starts_with(&old),
            "The source contains the open catalog; move the data-only folder instead"
        );
        ensure!(
            !old.join(".git").exists(),
            "A source move cannot move a Git checkout"
        );
        ensure!(
            !destination.exists(),
            "Destination already exists; no files were changed"
        );
        let absolute = if destination.is_absolute() {
            destination.to_owned()
        } else {
            std::env::current_dir()?.join(destination)
        };
        let parent = absolute
            .parent()
            .context("Destination needs a parent")?
            .canonicalize()
            .context("Destination parent must already exist")?;
        let new = parent.join(
            absolute
                .file_name()
                .context("Destination needs a folder name")?,
        );
        ensure!(
            !new.starts_with(&old) && !old.starts_with(&new),
            "Source and destination cannot contain each other"
        );
        for other in self.summary()?.sources {
            if other.name == source {
                continue;
            }
            let other_root = self.root(&other.name)?.canonicalize()?;
            ensure!(
                !old.starts_with(&other_root) && !other_root.starts_with(&old),
                "Source overlaps {}; repoint overlapping sources separately",
                other.name
            );
        }
        ensure!(
            self.check()?.is_empty(),
            "Fix catalog validation errors before moving folders"
        );
        let stored = pathdiff::diff_paths(&new, self.path.parent().unwrap()).unwrap_or(new.clone());
        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        let update = self.conn.execute(
            "UPDATE sources SET root_path=?1 WHERE name=?2",
            params![stored.to_str().context("Use UTF-8 paths")?, source],
        );
        if let Err(error) = update {
            self.conn.execute_batch("ROLLBACK")?;
            return Err(error.into());
        }
        if let Err(error) = fs::rename(&old, &new) {
            self.conn.execute_batch("ROLLBACK")?;
            return Err(error).context(
                "Move failed. For another filesystem, copy the folder first and use Repoint.",
            );
        }
        if let Err(error) = self.conn.execute_batch("COMMIT") {
            let _ = self.conn.execute_batch("ROLLBACK");
            fs::rename(&new, &old).context(
                "SQL save and filesystem rollback failed; recover the source from the destination",
            )?;
            return Err(error.into());
        }
        Ok(())
    }

    pub fn summary(&self) -> Result<Summary> {
        let mut summary = Summary::default();
        for (index, split) in SPLITS.iter().enumerate() {
            summary.splits[index] = self.conn.query_row(
                "SELECT COUNT(*) FROM dataset WHERE split=?1",
                [split],
                |r| r.get(0),
            )?;
        }
        summary.groups = self
            .conn
            .query_row("SELECT COUNT(*) FROM groups", [], |r| r.get(0))?;
        let mut query = self.conn.prepare("SELECT name,root_path,collected_at,notes,images,min_width,max_width,min_height,max_height,train,val,test FROM source_summary ORDER BY collected_at DESC,name")?;
        summary.sources = query
            .query_map([], |r| {
                let min: Option<u32> = r.get(5)?;
                let resolution = match min {
                    None => "No images".to_owned(),
                    Some(w) => format!(
                        "{}–{} × {}–{} px",
                        w,
                        r.get::<_, u32>(6)?,
                        r.get::<_, u32>(7)?,
                        r.get::<_, u32>(8)?
                    ),
                };
                Ok(Source {
                    name: r.get(0)?,
                    root: r.get(1)?,
                    collected: r.get(2)?,
                    notes: r.get(3)?,
                    images: r.get(4)?,
                    resolution,
                    splits: [r.get(9)?, r.get(10)?, r.get(11)?],
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        let mut query = self.conn.prepare("SELECT l.id,l.name,l.color,COUNT(sl.sample_id) FROM labels l LEFT JOIN sample_labels sl ON sl.label_id=l.id GROUP BY l.id ORDER BY l.id")?;
        summary.labels = query
            .query_map([], |r| {
                let hex: String = r.get(2)?;
                let color =
                    [1, 3, 5].map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap_or(128));
                Ok(Label {
                    id: r.get(0)?,
                    name: r.get(1)?,
                    color,
                    images: r.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<_>>()?;
        Ok(summary)
    }

    pub fn page(&self, filter: &Filter, offset: usize) -> Result<(usize, Vec<Sample>)> {
        let clause = " WHERE (?1='' OR split=?1) AND (?2='' OR source=?2)
            AND (?3='' OR instr(lower(image_path || source || group_name), lower(?3)) > 0)
            AND (?4=-1 OR EXISTS(SELECT 1 FROM sample_labels sl WHERE sl.sample_id=dataset.id AND sl.label_id=?4))";
        let values = params![
            filter.split,
            filter.source,
            filter.search,
            filter.label.unwrap_or(-1)
        ];
        let count = self.conn.query_row(
            &format!("SELECT COUNT(*) FROM dataset{clause}"),
            values,
            |r| r.get(0),
        )?;
        let mut query = self.conn.prepare(&format!("SELECT id,source,group_name,split,root_path,image_path,mask_path,collected_at,width,height FROM dataset{clause} ORDER BY id LIMIT ?5 OFFSET ?6"))?;
        let base = self.path.parent().unwrap();
        let items = query
            .query_map(
                params![
                    filter.split,
                    filter.source,
                    filter.search,
                    filter.label.unwrap_or(-1),
                    PAGE_SIZE,
                    offset
                ],
                |r| {
                    Ok(Sample {
                        id: r.get(0)?,
                        source: r.get(1)?,
                        group: r.get(2)?,
                        split: r.get(3)?,
                        root: base.join(r.get::<_, String>(4)?),
                        image: r.get(5)?,
                        mask: r.get(6)?,
                        collected: r.get(7)?,
                        width: r.get(8)?,
                        height: r.get(9)?,
                    })
                },
            )?
            .collect::<rusqlite::Result<_>>()?;
        Ok((count, items))
    }

    pub fn query(&self, sql: &str) -> Result<QueryResult> {
        let conn = Connection::open_with_flags(&self.path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
        let start = std::time::Instant::now();
        conn.progress_handler(
            10_000,
            Some(move || start.elapsed() > Duration::from_secs(2)),
        );
        let mut statement = conn.prepare(sql)?;
        ensure!(
            statement.readonly() && statement.column_count() > 0,
            "Use Run query for SELECT statements; Apply updates saves location/split edits"
        );
        let columns: Vec<String> = statement
            .column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut cursor = statement.query([])?;
        let mut rows = Vec::new();
        let mut truncated = false;
        while let Some(row) = cursor.next()? {
            if rows.len() == 200 {
                truncated = true;
                break;
            }
            rows.push(
                (0..columns.len())
                    .map(|i| match row.get_ref(i) {
                        Ok(ValueRef::Null) => "NULL".to_owned(),
                        Ok(ValueRef::Integer(n)) => n.to_string(),
                        Ok(ValueRef::Real(n)) => n.to_string(),
                        Ok(ValueRef::Text(s)) => String::from_utf8_lossy(s).into_owned(),
                        Ok(ValueRef::Blob(b)) => format!("[{} bytes]", b.len()),
                        Err(e) => e.to_string(),
                    })
                    .collect(),
            );
        }
        Ok(QueryResult {
            columns,
            rows,
            truncated,
        })
    }

    /// SQL edits are intentionally limited to locations, collection notes, and group splits.
    pub fn apply_sql(&self, sql: &str) -> Result<()> {
        self.conn.execute_batch("BEGIN IMMEDIATE")?;
        self.conn.authorizer(Some(edit_authorizer));
        let start = std::time::Instant::now();
        self.conn.progress_handler(
            10_000,
            Some(move || start.elapsed() > Duration::from_secs(2)),
        );
        let result = self.conn.execute_batch(sql);
        self.conn.progress_handler(0, None::<fn() -> bool>);
        self.conn
            .authorizer(None::<fn(hooks::AuthContext<'_>) -> hooks::Authorization>);
        let result = result.map_err(anyhow::Error::from).and_then(|()| {
            let errors = self.check()?;
            ensure!(errors.is_empty(), "Update rejected:\n{}", errors.join("\n"));
            Ok(())
        });
        match result {
            Ok(()) => {
                self.conn.execute_batch("COMMIT")?;
                Ok(())
            }
            Err(error) => {
                self.conn.execute_batch("ROLLBACK")?;
                Err(error)
            }
        }
    }

    pub fn schema(&self) -> Result<String> {
        let mut statement = self.conn.prepare("SELECT sql FROM sqlite_schema WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' ORDER BY type,name")?;
        let statements = statement
            .query_map([], |r| r.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(statements.join(";\n\n") + ";")
    }

    pub fn save_snapshot(&self, path: &Path) -> Result<()> {
        ensure!(
            path.extension().and_then(|s| s.to_str()) == Some("sql"),
            "Save snapshots as a .sql file"
        );
        ensure!(
            path.file_name().and_then(|s| s.to_str()) != Some("catalog.sql"),
            "catalog.sql is the application schema; save as catalog.snapshot.sql instead"
        );
        if path.exists() {
            ensure!(
                path.canonicalize()? != self.path,
                "Cannot overwrite the open catalog"
            );
        }
        fs::write(path, self.snapshot()?).with_context(|| format!("Writing {}", path.display()))
    }

    pub fn snapshot(&self) -> Result<String> {
        let tx = self.conn.unchecked_transaction()?;
        let mut output = String::from(
            "-- BEE detector metadata snapshot; paths are relative to the catalog directory.\nPRAGMA foreign_keys=OFF;\nBEGIN TRANSACTION;\n",
        );
        for table in ["sources", "groups", "labels", "samples", "sample_labels"] {
            let create: String = tx.query_row(
                "SELECT sql FROM sqlite_schema WHERE name=?1",
                [table],
                |r| r.get(0),
            )?;
            output.push_str(&format!("{create};\n"));
            let mut info = tx.prepare(&format!("PRAGMA table_info(\"{table}\")"))?;
            let columns = info
                .query_map([], |r| r.get::<_, String>(1))?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            let quoted = columns
                .iter()
                .map(|n| format!("quote(\"{n}\")"))
                .collect::<Vec<_>>()
                .join(",");
            let mut rows = tx.prepare(&format!("SELECT {quoted} FROM \"{table}\" ORDER BY 1"))?;
            let mut cursor = rows.query([])?;
            while let Some(row) = cursor.next()? {
                let values = (0..columns.len())
                    .map(|i| row.get::<_, String>(i))
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                output.push_str(&format!(
                    "INSERT INTO \"{table}\" VALUES ({});\n",
                    values.join(",")
                ));
            }
        }
        let mut objects = tx.prepare("SELECT sql FROM sqlite_schema WHERE type IN ('index','view') AND sql IS NOT NULL ORDER BY type,name")?;
        for statement in objects.query_map([], |r| r.get::<_, String>(0))? {
            output.push_str(&format!("{};\n", statement?));
        }
        output.push_str("PRAGMA user_version=2;\nCOMMIT;\nPRAGMA foreign_keys=ON;\n");
        Ok(output)
    }

    pub fn check(&self) -> Result<Vec<String>> {
        let mut errors = Vec::new();
        let broken: usize =
            self.conn
                .query_row("SELECT COUNT(*) FROM pragma_foreign_key_check", [], |r| {
                    r.get(0)
                })?;
        if broken > 0 {
            errors.push(format!("{broken} foreign key violations"));
        }
        for source in self.summary()?.sources {
            if let Err(error) = validate_source(&source.name, &source.collected) {
                errors.push(error.to_string());
            }
            if !self.root(&source.name)?.is_dir() {
                errors.push(format!("{}: source folder is missing", source.name));
            }
        }
        let mut query = self.conn.prepare("SELECT id,root_path,image_path,mask_path,image_sha256,mask_sha256,width,height FROM dataset ORDER BY id")?;
        let mut rows = query.query([])?;
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let root = self.path.parent().unwrap().join(row.get::<_, String>(1)?);
            let result = (|| -> Result<()> {
                let image = resolve(&root, Path::new(&row.get::<_, String>(2)?))?;
                let mask = resolve(&root, Path::new(&row.get::<_, String>(3)?))?;
                let actual = inspect(&self.conn, &image, &mask)?;
                ensure!(
                    actual.image_hash == row.get::<_, String>(4)?
                        && actual.mask_hash == row.get::<_, String>(5)?,
                    "File content changed; review and re-import"
                );
                ensure!(
                    actual.width == row.get::<_, u32>(6)?
                        && actual.height == row.get::<_, u32>(7)?,
                    "Image resolution changed"
                );
                let mut counts = self.conn.prepare("SELECT label_id,pixels FROM sample_labels WHERE sample_id=?1 ORDER BY label_id")?;
                let stored = counts
                    .query_map([id], |r| Ok((r.get::<_, u8>(0)?, r.get::<_, u64>(1)?)))?
                    .collect::<rusqlite::Result<BTreeMap<_, _>>>()?;
                ensure!(stored == actual.counts, "Mask label counts changed");
                Ok(())
            })();
            if let Err(error) = result {
                errors.push(format!("Sample {id}: {error:#}"));
            }
        }
        Ok(errors)
    }
}

fn edit_authorizer(context: hooks::AuthContext<'_>) -> hooks::Authorization {
    use hooks::{AuthAction, Authorization};
    match context.action {
        AuthAction::Select | AuthAction::Read { .. } | AuthAction::Function { .. } => {
            Authorization::Allow
        }
        AuthAction::Update {
            table_name: "sources",
            column_name: "root_path" | "notes",
            ..
        }
        | AuthAction::Update {
            table_name: "groups",
            column_name: "split",
            ..
        }
        | AuthAction::Update {
            table_name: "samples",
            column_name: "image_path" | "mask_path",
            ..
        } => Authorization::Allow,
        _ => Authorization::Deny,
    }
}

pub fn resolve(root: &Path, path: &Path) -> Result<PathBuf> {
    let root = root.canonicalize().context("Source folder is missing")?;
    let candidate = root
        .join(path)
        .canonicalize()
        .with_context(|| format!("File missing: {}", path.display()))?;
    ensure!(candidate.starts_with(&root), "Path escapes the source root");
    ensure!(
        candidate.is_file(),
        "Expected a file: {}",
        candidate.display()
    );
    Ok(candidate)
}

fn validate_source(name: &str, collected: &str) -> Result<()> {
    let parts: Vec<_> = name.split("__").collect();
    ensure!(
        parts.len() == 3,
        "Name sources <hive>__<YYYY-MM-DD>__<camera>"
    );
    for slug in [parts[0], parts[2]] {
        ensure!(
            !slug.is_empty()
                && slug
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
            "Hive and camera names use lowercase letters, digits, and hyphens"
        );
    }
    let date = NaiveDate::parse_from_str(collected, "%Y-%m-%d")
        .context("Collection date must be YYYY-MM-DD")?;
    ensure!(
        date.format("%Y-%m-%d").to_string() == collected && parts[1] == collected,
        "Source name must contain its collection date"
    );
    Ok(())
}

pub fn read_mask(path: &Path) -> Result<Mask> {
    let mut decoder = png::Decoder::new(BufReader::new(fs::File::open(path)?));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info()?;
    let info = reader.info();
    ensure!(
        info.bit_depth == png::BitDepth::Eight
            && matches!(
                info.color_type,
                png::ColorType::Grayscale | png::ColorType::Indexed
            ),
        "Mask must be an 8-bit grayscale or indexed PNG"
    );
    let mut ids = vec![0; reader.output_buffer_size()];
    let output = reader.next_frame(&mut ids)?;
    ids.truncate(output.buffer_size());
    ensure!(
        ids.len() == output.width as usize * output.height as usize,
        "Mask must contain one class ID per pixel"
    );
    Ok(Mask {
        width: output.width,
        height: output.height,
        ids,
    })
}

fn hash(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut block = vec![0; 1024 * 1024];
    loop {
        let count = file.read(&mut block)?;
        if count == 0 {
            break;
        }
        digest.update(&block[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn inspect(conn: &Connection, image: &Path, mask: &Path) -> Result<Inspected> {
    let frame = image::open(image).with_context(|| format!("Reading {}", image.display()))?;
    ensure!(
        frame.color().channel_count() == 3,
        "Input images must have exactly three RGB channels"
    );
    let mask_ids = read_mask(mask)?;
    ensure!(
        (frame.width(), frame.height()) == (mask_ids.width, mask_ids.height),
        "Image and mask resolutions differ"
    );
    let mut counts = BTreeMap::new();
    for id in mask_ids.ids {
        *counts.entry(id).or_insert(0_u64) += 1;
    }
    ensure!(
        counts.keys().any(|id| *id != 255),
        "Mask is entirely ignored; annotate it before importing"
    );
    for id in counts.keys() {
        let exists: bool = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM labels WHERE id=?1)",
            [id],
            |r| r.get(0),
        )?;
        ensure!(exists, "Unknown mask label {id}");
    }
    Ok(Inspected {
        width: frame.width(),
        height: frame.height(),
        image_hash: hash(image)?,
        mask_hash: hash(mask)?,
        counts,
    })
}

fn register_pair(
    conn: &Connection,
    root: &Path,
    source: &str,
    image: &Path,
    mask: &Path,
    split: &str,
    group: &str,
) -> Result<i64> {
    ensure!(
        SPLITS.contains(&split) && !group.trim().is_empty(),
        "A valid split and collection group are required"
    );
    let image = resolve(root, image)?;
    let mask = resolve(root, mask)?;
    let metadata = inspect(conn, &image, &mask)?;
    let image_path = image
        .strip_prefix(root)?
        .to_str()
        .context("Use UTF-8 filenames")?;
    let mask_path = mask
        .strip_prefix(root)?
        .to_str()
        .context("Use UTF-8 filenames")?;
    conn.execute(
        "INSERT OR IGNORE INTO groups(name,split) VALUES (?1,?2)",
        params![group, split],
    )?;
    let current: String =
        conn.query_row("SELECT split FROM groups WHERE name=?1", [group], |r| {
            r.get(0)
        })?;
    ensure!(
        current == split,
        "Group leakage: {group} already belongs to {current}"
    );
    let old_group: Option<String> = conn
        .query_row(
            "SELECT group_name FROM samples WHERE source=?1 AND image_path=?2",
            params![source, image_path],
            |r| r.get(0),
        )
        .optional()?;
    ensure!(
        old_group.as_deref().is_none_or(|g| g == group),
        "Re-import cannot change an image's group"
    );
    conn.execute("INSERT INTO samples(source,group_name,image_path,mask_path,image_sha256,mask_sha256,width,height)
        VALUES (?1,?2,?3,?4,?5,?6,?7,?8) ON CONFLICT(source,image_path) DO UPDATE SET
        mask_path=excluded.mask_path,image_sha256=excluded.image_sha256,mask_sha256=excluded.mask_sha256,width=excluded.width,height=excluded.height",
        params![source,group,image_path,mask_path,metadata.image_hash,metadata.mask_hash,metadata.width,metadata.height])?;
    let id = conn.query_row(
        "SELECT id FROM samples WHERE source=?1 AND image_path=?2",
        params![source, image_path],
        |r| r.get(0),
    )?;
    conn.execute("DELETE FROM sample_labels WHERE sample_id=?1", [id])?;
    for (label, pixels) in metadata.counts {
        conn.execute(
            "INSERT INTO sample_labels VALUES (?1,?2,?3)",
            params![id, label, pixels],
        )?;
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    const SOURCE: &str = "hive-07__2026-06-12__camera-a";

    fn fixture() -> (TempDir, Catalog, PathBuf) {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("any-existing-layout");
        fs::create_dir_all(&root).unwrap();
        let db = Catalog::open(&temp.path().join("data/catalog.sqlite3")).unwrap();
        db.add_source(SOURCE, &root, "2026-06-12", "Beekeeper collection")
            .unwrap();
        (temp, db, root)
    }

    fn pair(root: &Path, name: &str, color: u8, label: u8) {
        let image = image::RgbImage::from_pixel(31, 17, image::Rgb([color, 180, 50]));
        image.save(root.join(format!("{name}.png"))).unwrap();
        image::GrayImage::from_pixel(31, 17, image::Luma([label]))
            .save(root.join(format!("{name}.mask.png")))
            .unwrap();
    }

    #[test]
    fn sql_drives_splits_and_arbitrary_paths() {
        let (_temp, mut db, root) = fixture();
        fs::create_dir(root.join("existing-images")).unwrap();
        fs::create_dir(root.join("existing-annotations")).unwrap();
        pair(&root, "old-name", 12, 3);
        fs::rename(
            root.join("old-name.png"),
            root.join("existing-images/bee.png"),
        )
        .unwrap();
        fs::rename(
            root.join("old-name.mask.png"),
            root.join("existing-annotations/bee.png"),
        )
        .unwrap();
        let id = db
            .register(
                SOURCE,
                Path::new("existing-images/bee.png"),
                Path::new("existing-annotations/bee.png"),
                "train",
                "hive-07",
            )
            .unwrap();
        let summary = db.summary().unwrap();
        assert_eq!(summary.splits, [1, 0, 0]);
        assert_eq!(summary.sources[0].collected, "2026-06-12");
        assert_eq!(summary.sources[0].resolution, "31–31 × 17–17 px");
        assert_eq!(db.move_group("hive-07", "test").unwrap(), 1);
        let (total, rows) = db
            .page(
                &Filter {
                    split: "test".into(),
                    ..Default::default()
                },
                0,
            )
            .unwrap();
        assert_eq!(total, 1);
        assert_eq!(rows[0].id, id);
        assert_eq!(rows[0].width, 31);
        assert!(root.join("existing-images/bee.png").exists());
        assert!(!root.join("test").exists());
        assert!(db.check().unwrap().is_empty());
    }

    #[test]
    fn imports_preserve_ids_and_reject_group_leakage() {
        let (_temp, mut db, root) = fixture();
        pair(&root, "frame-000001", 12, 1);
        assert_eq!(db.import(SOURCE, "train", "hive-07").unwrap(), 1);
        let id = db.page(&Filter::default(), 0).unwrap().1[0].id;
        db.import(SOURCE, "train", "hive-07").unwrap();
        assert_eq!(db.page(&Filter::default(), 0).unwrap().1[0].id, id);
        let error = db.import(SOURCE, "val", "hive-07").unwrap_err();
        assert!(error.to_string().contains("Group leakage"));
        assert_eq!(db.summary().unwrap().splits, [1, 0, 0]);
    }

    #[test]
    fn import_is_atomic_on_bad_masks_and_duplicates() {
        let (_temp, mut db, root) = fixture();
        pair(&root, "a", 12, 1);
        pair(&root, "b", 13, 77);
        assert!(db.import(SOURCE, "train", "hive-07").is_err());
        assert_eq!(db.summary().unwrap().splits, [0, 0, 0]);
        pair(&root, "b", 12, 1);
        assert!(db.import(SOURCE, "train", "hive-07").is_err());
        assert_eq!(db.summary().unwrap().splits, [0, 0, 0]);
        fs::remove_file(root.join("b.mask.png")).unwrap();
        assert!(db.import(SOURCE, "train", "hive-07").is_err());
        assert_eq!(db.summary().unwrap().groups, 0);
    }

    #[test]
    fn masks_reject_wrong_dimensions_rgb_and_entirely_ignored() {
        let (_temp, mut db, root) = fixture();
        pair(&root, "a", 12, 255);
        assert!(
            db.import(SOURCE, "train", "hive-07")
                .unwrap_err()
                .to_string()
                .contains("entirely ignored")
        );
        image::GrayImage::from_pixel(2, 2, image::Luma([1]))
            .save(root.join("a.mask.png"))
            .unwrap();
        assert!(
            db.import(SOURCE, "train", "hive-07")
                .unwrap_err()
                .to_string()
                .contains("resolutions differ")
        );
        image::RgbImage::new(31, 17)
            .save(root.join("a.mask.png"))
            .unwrap();
        assert!(
            db.import(SOURCE, "train", "hive-07")
                .unwrap_err()
                .to_string()
                .contains("8-bit")
        );
    }

    #[test]
    fn palette_masks_keep_indices_and_ignore_pixels() {
        let (_temp, mut db, root) = fixture();
        pair(&root, "a", 12, 1);
        let file = fs::File::create(root.join("a.mask.png")).unwrap();
        let mut encoder = png::Encoder::new(file, 31, 17);
        encoder.set_color(png::ColorType::Indexed);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_palette(vec![255, 255, 255, 5, 200, 90]);
        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&vec![1; 31 * 17]).unwrap();
        drop(writer);
        db.import(SOURCE, "train", "hive-07").unwrap();
        assert_eq!(
            read_mask(&root.join("a.mask.png")).unwrap().ids,
            vec![1; 31 * 17]
        );
        assert_eq!(
            db.summary()
                .unwrap()
                .labels
                .iter()
                .find(|l| l.id == 1)
                .unwrap()
                .images,
            1
        );
    }

    #[test]
    fn high_range_rgb_is_accepted_without_changing_source_pixels() {
        let (_temp, mut db, root) = fixture();
        let pixels = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::from_pixel(
            31,
            17,
            image::Rgb([4096, 32768, 65535]),
        );
        pixels.save(root.join("rgb16.png")).unwrap();
        image::GrayImage::from_pixel(31, 17, image::Luma([1]))
            .save(root.join("rgb16.mask.png"))
            .unwrap();
        let before = hash(&root.join("rgb16.png")).unwrap();
        db.import(SOURCE, "train", "hive-07").unwrap();
        assert_eq!(before, hash(&root.join("rgb16.png")).unwrap());
        assert!(db.check().unwrap().is_empty());
        let hdr = image::ImageBuffer::<image::Rgb<f32>, Vec<f32>>::from_pixel(
            31,
            17,
            image::Rgb([0.125, 1.5, 12.0]),
        );
        hdr.save(root.join("hdr.tiff")).unwrap();
        image::GrayImage::from_pixel(31, 17, image::Luma([1]))
            .save(root.join("hdr.mask.png"))
            .unwrap();
        let hdr_hash = hash(&root.join("hdr.tiff")).unwrap();
        db.import(SOURCE, "train", "hive-07").unwrap();
        assert_eq!(hdr_hash, hash(&root.join("hdr.tiff")).unwrap());
        assert_eq!(db.summary().unwrap().splits, [2, 0, 0]);
        image::GrayImage::from_pixel(31, 17, image::Luma([20]))
            .save(root.join("gray.png"))
            .unwrap();
        assert!(
            db.register(
                SOURCE,
                Path::new("gray.png"),
                Path::new("rgb16.mask.png"),
                "train",
                "hive-07"
            )
            .unwrap_err()
            .to_string()
            .contains("three RGB channels")
        );
    }

    #[test]
    fn sql_updates_are_atomic_and_protect_schema() {
        let (_temp, mut db, root) = fixture();
        pair(&root, "a", 12, 1);
        db.import(SOURCE, "train", "hive-07").unwrap();
        db.apply_sql("UPDATE groups SET split='val' WHERE name='hive-07'; UPDATE sources SET notes='reviewed';").unwrap();
        assert_eq!(db.summary().unwrap().splits, [0, 1, 0]);
        assert_eq!(db.summary().unwrap().sources[0].notes, "reviewed");
        assert!(db.apply_sql("UPDATE groups SET split='test'; UPDATE sources SET root_path='/missing/bee-fixture';").is_err());
        assert_eq!(db.summary().unwrap().splits, [0, 1, 0]);
        for sql in [
            "DROP TABLE samples",
            "DELETE FROM samples",
            "UPDATE labels SET id=77",
            "PRAGMA foreign_keys=OFF",
            "COMMIT",
            "ATTACH DATABASE ':memory:' AS extra",
        ] {
            assert!(db.apply_sql(sql).is_err(), "{sql}");
        }
        assert!(db.check().unwrap().is_empty());
        assert!(db.query("DELETE FROM samples").is_err());
        assert_eq!(
            db.query("SELECT COUNT(*) FROM dataset").unwrap().rows[0][0],
            "1"
        );
    }

    #[test]
    fn sql_can_remap_individual_files_without_layout_constraints() {
        let (_temp, mut db, root) = fixture();
        pair(&root, "a", 12, 1);
        db.import(SOURCE, "train", "hive-07").unwrap();
        fs::rename(root.join("a.png"), root.join("renamed.png")).unwrap();
        fs::rename(root.join("a.mask.png"), root.join("other-mask.png")).unwrap();
        db.apply_sql("UPDATE samples SET image_path='renamed.png', mask_path='other-mask.png';")
            .unwrap();
        assert!(db.check().unwrap().is_empty());
    }

    #[test]
    fn source_move_updates_sql_and_refuses_overwrites() {
        let (temp, mut db, root) = fixture();
        pair(&root, "a", 12, 1);
        db.import(SOURCE, "train", "hive-07").unwrap();
        let destination = temp.path().join("moved-collection");
        db.move_source(SOURCE, &destination).unwrap();
        assert!(!root.exists());
        assert!(destination.join("a.png").exists());
        assert_eq!(
            db.root(SOURCE).unwrap().canonicalize().unwrap(),
            destination
        );
        assert!(db.check().unwrap().is_empty());
        let occupied = temp.path().join("occupied");
        fs::create_dir(&occupied).unwrap();
        assert!(db.move_source(SOURCE, &occupied).is_err());
        assert!(destination.join("a.png").exists());
    }

    #[test]
    fn repoint_verifies_hashes_and_keeps_old_pointer_on_failure() {
        let (temp, mut db, root) = fixture();
        pair(&root, "a", 12, 1);
        assert!(db.relocate("unknown-source", &root).is_err());
        db.import(SOURCE, "train", "hive-07").unwrap();
        let copy = temp.path().join("copy");
        fs::create_dir(&copy).unwrap();
        pair(&copy, "a", 30, 1);
        assert!(db.relocate(SOURCE, &copy).is_err());
        assert_eq!(db.root(SOURCE).unwrap().canonicalize().unwrap(), root);
        fs::copy(root.join("a.png"), copy.join("a.png")).unwrap();
        db.relocate(SOURCE, &copy).unwrap();
        assert_eq!(db.root(SOURCE).unwrap().canonicalize().unwrap(), copy);
    }

    #[test]
    fn relative_roots_and_sql_snapshot_roundtrip() {
        let (temp, mut db, root) = fixture();
        pair(&root, "a", 12, 3);
        db.import(SOURCE, "train", "hive-07").unwrap();
        let snapshot = db.snapshot().unwrap();
        assert!(db.save_snapshot(&db.path).is_err());
        assert!(db.save_snapshot(&temp.path().join("catalog.sql")).is_err());
        let restored_path = temp.path().join("data/restored.sqlite3");
        let restored = Connection::open(&restored_path).unwrap();
        restored.execute_batch(&snapshot).unwrap();
        drop(restored);
        let restored = Catalog::open(&restored_path).unwrap();
        assert_eq!(restored.summary().unwrap().splits, [1, 0, 0]);
        assert!(restored.check().unwrap().is_empty());
        assert!(
            restored.summary().unwrap().sources[0]
                .root
                .starts_with("..")
        );
    }

    #[test]
    fn names_dates_and_path_escape_are_checked() {
        let (temp, mut db, root) = fixture();
        pair(&root, "a", 12, 1);
        assert!(db.add_source("bad-name", &root, "2026-06-12", "").is_err());
        assert!(
            db.add_source("hive__2026-02-30__camera", &root, "2026-02-30", "")
                .is_err()
        );
        assert!(
            db.add_source("hive__2026-06-13__camera", &root, "2026-06-12", "")
                .is_err()
        );
        image::RgbImage::new(31, 17)
            .save(temp.path().join("outside.png"))
            .unwrap();
        assert!(
            db.register(
                SOURCE,
                Path::new("../outside.png"),
                Path::new("a.mask.png"),
                "train",
                "hive-07"
            )
            .is_err()
        );
    }

    #[test]
    fn reopening_a_versioned_catalog_does_not_rewrite_it() {
        let (_temp, db, _root) = fixture();
        let before = hash(&db.path).unwrap();
        let path = db.path.clone();
        drop(db);
        let reopened = Catalog::open(&path).unwrap();
        assert_eq!(hash(&reopened.path).unwrap(), before);
    }

    #[test]
    fn labels_text_matches_the_sql_vocabulary() {
        let (_temp, db, _root) = fixture();
        let expected: Vec<(i64, String)> = include_str!("../data/labels.txt")
            .lines()
            .filter(|line| !line.starts_with('#') && !line.is_empty())
            .map(|line| {
                let (id, name) = line.split_once(' ').unwrap();
                (id.parse().unwrap(), name.to_owned())
            })
            .collect();
        let actual: Vec<_> = db
            .summary()
            .unwrap()
            .labels
            .into_iter()
            .map(|label| (label.id, label.name))
            .collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn pagination_and_filtering_cover_the_whole_catalog() {
        let (_temp, mut db, root) = fixture();
        for i in 0..27 {
            pair(
                &root,
                &format!("frame-{i:04}"),
                i as u8,
                if i % 2 == 0 { 1 } else { 3 },
            );
        }
        db.import(SOURCE, "train", "hive-07").unwrap();
        let (count, first) = db.page(&Filter::default(), 0).unwrap();
        assert_eq!(count, 27);
        assert_eq!(first.len(), PAGE_SIZE);
        assert_eq!(db.page(&Filter::default(), PAGE_SIZE).unwrap().1.len(), 3);
        assert_eq!(
            db.page(
                &Filter {
                    label: Some(3),
                    ..Default::default()
                },
                0
            )
            .unwrap()
            .0,
            13
        );
        assert_eq!(
            db.page(
                &Filter {
                    search: "frame-0001".into(),
                    ..Default::default()
                },
                0
            )
            .unwrap()
            .0,
            1
        );
    }
}
