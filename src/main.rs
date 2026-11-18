mod catalog;
mod view;

use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Bee and hive-health segmentation dataset tools")]
struct Args {
    #[arg(long, default_value = "data/catalog.sqlite3", global = true)]
    catalog: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Open the native Rust data explorer.
    BeeData,
    /// Create the SQLite catalog.
    Init,
    /// Register a collection folder: <hive>__<YYYY-MM-DD>__<camera>.
    Source {
        name: String,
        #[arg(long)]
        path: PathBuf,
        #[arg(long)]
        collected: String,
        #[arg(long, default_value = "")]
        notes: String,
    },
    /// Import image + <stem>.mask.png pairs from a source folder, recursively.
    Import {
        source: String,
        #[arg(long, value_parser = ["train", "val", "test"])]
        split: String,
        /// Keep a hive/visit/video group in one split. Defaults to the source name.
        #[arg(long)]
        group: Option<String>,
    },
    /// Register explicitly mapped files with any layout inside a source root.
    Register {
        source: String,
        #[arg(long)]
        image: PathBuf,
        #[arg(long)]
        mask: PathBuf,
        #[arg(long, value_parser = ["train", "val", "test"])]
        split: String,
        #[arg(long)]
        group: Option<String>,
    },
    /// Repoint one source after moving its files; verify hashes before saving.
    Relocate { source: String, path: PathBuf },
    /// Change logical split membership without moving any files.
    MoveGroup {
        group: String,
        #[arg(long, value_parser = ["train", "val", "test"])]
        split: String,
    },
    /// Verify files, dimensions, label IDs, hashes, and foreign keys.
    Check,
    /// Run a read-only SQL query against the current catalog.
    Sql {
        query: String,
        /// Apply location / split updates in a validated transaction.
        #[arg(long)]
        apply: bool,
    },
    /// Export a portable SQL document with all metadata and source locations.
    Snapshot { path: PathBuf },
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut db = catalog::Catalog::open(&args.catalog)?;
    match args.command {
        Command::BeeData => view::run(db)?,
        Command::Init => println!("Catalog ready: {}", db.path.display()),
        Command::Source {
            name,
            path,
            collected,
            notes,
        } => {
            db.add_source(&name, &path, &collected, &notes)?;
            println!("Registered {name}");
        }
        Command::Import {
            source,
            split,
            group,
        } => {
            let count = db.import(&source, &split, group.as_deref().unwrap_or(&source))?;
            println!("Indexed {count} pairs in {split}");
        }
        Command::Register {
            source,
            image,
            mask,
            split,
            group,
        } => {
            let id = db.register(
                &source,
                &image,
                &mask,
                &split,
                group.as_deref().unwrap_or(&source),
            )?;
            println!("Indexed sample {id}");
        }
        Command::Relocate { source, path } => {
            db.relocate(&source, &path)?;
            println!("Updated the location of {source}");
        }
        Command::MoveGroup { group, split } => {
            println!(
                "Reassigned {} images to {split}",
                db.move_group(&group, &split)?
            );
        }
        Command::Check => {
            let errors = db.check()?;
            if !errors.is_empty() {
                bail!("{}", errors.join("\n"));
            }
            println!("All cataloged pairs are valid");
        }
        Command::Sql { query, apply: true } => {
            db.apply_sql(&query)?;
            println!("Catalog updates saved");
        }
        Command::Sql {
            query,
            apply: false,
        } => {
            let result = db.query(&query)?;
            println!("{}", result.columns.join("\t"));
            for row in result.rows {
                println!("{}", row.join("\t"));
            }
            if result.truncated {
                println!("Showing the first 200 rows");
            }
        }
        Command::Snapshot { path } => {
            db.save_snapshot(&path)?;
            println!("Saved {}", path.display());
        }
    }
    Ok(())
}
