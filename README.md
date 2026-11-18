![BEE detector — Bee and hive segmentation dataset](banner.png)

**BEE detector** is a dataset repository for studying bees and hive health through
image segmentation. It supports research and tools that help beekeepers
understand and care for their hives.

Explore the dataset:

```sh
cargo run --release -- bee-data
```

Browse images in a bento grid, inspect sources, collection dates and resolutions,
and move source folders or update their SQL locations. SQLite defines the train,
val and test splits independently of the folder layout. The attached SQL catalog
holds only paths and metadata; Git LFS versions the images and masks.
Labels cover workers, drones, queens, pollen, honey, pests and visible disease
signs, listed in [data/labels.txt](data/labels.txt).
[GVAT](https://github.com/jldx-1/gvat) is used to label and review the images and
segmentation annotations. Setup and PyTorch examples follow below.

## Setup

Run commands from the repository root. The catalog and native viewer are Rust;
Python is only an optional PyTorch reader. Linux is the supported platform.

**Image input:** three-channel **RGB** (red, green, blue), with PyTorch batches
shaped `[B, 3, H, W]`. Standard 8-bit RGB, 16-bit RGB PNG/TIFF and floating-point
RGB TIFF can be stored without changing the source pixels. Use the reader's
`--high-range` option for 16-bit or high dynamic range (HDR) inputs.

**Segmentation format:** per-pixel semantic segmentation. Each image is paired
with a same-resolution, 8-bit grayscale or indexed PNG mask. Mask pixel values
are the class IDs in [data/labels.txt](data/labels.txt): `0` is background and
`255` is ignored. Masks contain no instance IDs.

### Start

Install Rust 1.88+, Git LFS, and graphics build libraries. On Ubuntu:

```sh
sudo apt-get install build-essential pkg-config \
  libxkbcommon-dev libwayland-dev libegl1-mesa-dev \
  libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev \
  libx11-xcb1 libxcursor1 libxi6 libxkbcommon-x11-0
git lfs install --local
git lfs pull
cargo run --release -- bee-data
```

The command builds and opens a native Linux window using X11 or Wayland.
The repository includes a small `data/catalog.sqlite3` metadata database; the app
can also initialize an empty catalog with `cargo run -- init`. No sample images or physical
train/val/test directories are created. There is no web or Python service.

```text
src/              Rust catalog and window; tests live alongside the code
data/             SQL schema, local SQLite catalog, and optional local data
Cargo.toml        Rust dependencies; Cargo.lock fixes their versions
pytorch.py        Optional, lazy PyTorch reader and runnable loading example
README.md         Description, setup, catalog workflow and PyTorch loading
LICENSE           MIT license
banner.png        Rounded-corner illustration, not training data
```

## Sources and naming

Use `<hive>__<YYYY-MM-DD>__<camera>`, for example
`hive-07__2026-06-12__camera-a`. Hive and camera names use lowercase letters,
digits and hyphens. Enter the collection date explicitly; file modification
times do not establish when an image was collected. Record provenance, usage
rights and inspection evidence in source notes or a release record.

Each source points to a folder that can have any name and live anywhere.
Image/mask paths are relative to that root. Automatic import pairs
`frame-000001.jpg` with `frame-000001.mask.png` in the same directory, including
nested folders. Explicit registration supports existing alternative layouts.

In **Sources**, register a name, date and folder. Open **Manage location / index**,
select a split and group, and choose **Index source**. The bento overview shows
all images, split counts, label coverage and masks. Inspect an image to see its
source, collection date, full resolution, group and exact paths.

Equivalent commands:

```sh
cargo run -- source hive-07__2026-06-12__camera-a --path /datasets/hive-07 --collected 2026-06-12 --notes 'Collection provenance'
cargo run -- import hive-07__2026-06-12__camera-a --split train --group hive-07
cargo run -- register hive-07__2026-06-12__camera-a --image photos/0001.jpg --mask labels/0001.png --split train --group hive-07
cargo run -- check
```

Register file paths relative to the source folder or as absolute paths inside
it. Import is transactional and checks dimensions, numeric IDs and SHA-256
hashes. It rejects duplicate image bytes, missing pairs and groups spread across
splits. Re-import preserves IDs and refreshes annotations; it does not prune
manually deleted files. Hashes do not detect near-duplicate or re-encoded frames.
Normalize image rotation before annotation.

## SQL and moving data

`data/catalog.sql` is the versioned schema and label vocabulary.
`data/catalog.sqlite3` is the live database:

| Table or view | Contents |
| --- | --- |
| `sources` | Collection name, folder, collection date, notes, registration time |
| `groups` | Hive/visit/video group and its train, val or test membership |
| `samples` | Exact image/mask mappings, hashes, resolution, indexing time |
| `labels`, `sample_labels` | Class names, colors and pixel counts per image |
| `dataset` | Joined image records, source locations, dates and logical splits |
| `source_summary` | Per-source counts and resolution ranges |

Splits are SQL fields. Assigning a group to `val` changes SQL regardless of the
physical folder name or disk. Choose whole groups appropriate to your evaluation:
use hive groups for unseen-hive evaluation; avoid splitting adjacent video frames
between train and test. Keep final test groups fixed.

In **SQL catalog**, browse the live Sources, Images, Groups and Labels tables,
run SELECT queries, and expand the actual schema. Query results display up to
200 rows; the image grid pages through the complete filtered dataset.

**Apply updates** saves supported UPDATE statements in a validated transaction:

```sql
UPDATE sources
SET root_path = '/datasets/new-location/hive-07'
WHERE name = 'hive-07__2026-06-12__camera-a';
UPDATE groups SET split = 'val' WHERE name = 'hive-07';
```

Editable fields are `sources.root_path`, `sources.notes`, `samples.image_path`,
`samples.mask_path`, and `groups.split`. The app checks files and hashes before
saving and rolls back invalid edits. Relative source roots resolve against the
catalog directory; image/mask paths resolve against the source root. Updating a
path repoints metadata after files have moved; SQL edits alone do not move bytes.

For a physical move, use **Sources → Manage location → Move folder + update SQL**.
It renames the whole collection on the same filesystem, saves its new root and
rolls back on handled failures. The destination's parent must exist. It refuses
overwrites, overlapping sources, moves containing the open database and Git
checkout moves. For another filesystem, copy the collection first and use
**Repoint catalog**, which verifies indexed files and hashes. Stop training or
concurrent file operations before moving data. Filesystem renames and SQL cannot
be one crash-proof transaction; retain backups for interrupted storage.

The small `data/catalog.sqlite3` database is versioned with the repository. It
contains paths and metrics, never image or mask bytes. Keep roots relative where
possible for portability. SQLite journal/lock sidecars are ignored. Use
**SQL catalog → Save SQL document**, or the command below, for a readable SQL
snapshot containing the same pointer metadata:

```sh
cargo run -- snapshot data/catalog.snapshot.sql
```

Restore a snapshot into an empty database with:

```sh
sqlite3 data/catalog.sqlite3 < data/catalog.snapshot.sql
```

Keep the snapshot beside the catalog so relative roots resolve the same. Moving
the whole checkout preserves relative roots; repoint external sources individually.
Git LFS versions files inside the checkout. External collections need their own
storage/versioning arrangement.

## Segmentation labels

The class IDs and names are also listed in [data/labels.txt](data/labels.txt).
Use [GVAT](https://github.com/jldx-1/gvat) to label and review images and
segmentation annotations. This catalog reads class-ID PNG masks; convert
annotations to that format with the matching IDs before importing them.

This dataset targets **semantic segmentation**: one class per visible pixel,
without instance IDs. Training IDs are 0–13; 255 is ignored.

| ID | Label |
| --- | --- |
| 0 | background |
| 1–3 | worker, drone, queen |
| 4–5 | pollen, honey |
| 6–7 | capped_brood, uncapped_brood |
| 8–10 | varroa_mite, small_hive_beetle, wax_moth_larva |
| 11–13 | deformed_wing, chalkbrood_suspected, foulbrood_suspected |
| 255 | ignore / uncertain |

Masks must be 8-bit grayscale or indexed PNGs matching the image's resolution.
Palette indices are IDs; RGB visualizations are not training masks. Annotate
visible regions. An identified pest, resource or disease-sign region takes
priority over its underlying bee/brood class; use 255 for unresolved boundaries
or identity. Entirely ignored masks must be annotated before import. Keep
published IDs stable; vocabulary changes require an explicit migration.

Disease signs require an experienced reviewer and do not confirm pathogens.
`deformed_wing` does not assert a viral diagnosis; `foulbrood_suspected` does not
distinguish American from European foulbrood. References:
[Penn State Extension](https://extension.psu.edu/a-quick-reference-guide-to-honey-bee-parasites-pests-predators-and-diseases)
and [NC State Extension](https://content.ces.ncsu.edu/disease-management-and-guidelines-for-the-honey-bee).
No real images, trained model or accuracy claim is bundled. Add only imagery
you have permission to share under compatible terms.

## PyTorch

Install your [PyTorch build](https://pytorch.org/get-started/locally/) and
`numpy Pillow opencv-python-headless`, then run `python3 pytorch.py`. The reader
snapshots SQL mappings once and opens no database handles in workers. It lazily
decodes each pair:

```python
import torch
from pytorch import make_loader, to_device

dataset, loader = make_loader('data/catalog.sqlite3', split='train', workers=4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for images, masks in loader:
    images, masks = to_device(images, masks, device)
    # images [B,3,H,W]; masks [B,H,W]; dataset.num_classes output channels
    # loss = torch.nn.CrossEntropyLoss(ignore_index=255)(model(images), masks)
```

For higher-range RGB, run `python3 pytorch.py --high-range` or pass
`high_range=True` to `make_loader`. That mode returns float32 RGB: unsigned
8-/16-bit inputs are scaled by 255/65535, while floating-point TIFF values retain
their range, including values above 1. `to_device` avoids normalizing those
values a second time. The default reader rejects high-range inputs rather than
silently converting them to 8-bit. Viewer previews are 8-bit; source files are
unchanged. No trained model is bundled; use a three-channel input encoder and
choose normalization appropriate to its training data.

Default CPU image queues hold uint8 tensors; high-range queues hold float32.
The loader uses pinned memory for CUDA, persistent workers and bounded
prefetching. Benchmark worker count, batch size and resolution;
use `--workers 0` for debugging. Apply your encoder's normalization after scaling.
The fixed-size reader resizes masks with nearest-neighbor interpolation. Any
geometric augmentation must transform images and masks together. Recreate loaders
after moving files or editing SQL. See [DataLoader](https://docs.pytorch.org/docs/stable/data.html)
and [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html).

## Checks and artwork

`cargo fmt --all -- --check`, `cargo clippy --all-targets -- -D warnings` and
`cargo test --locked` check formatting, lints and Rust behavior. Linux CI also
smoke-tests the native window on a virtual display and checks PyTorch loading.
All fixtures stay in temporary directories, outside the real `data/` directory.

The rounded banner is illustrative branding made with built-in image generation.
Its boxes do not describe the pixel-mask training format, and the artwork is not
training evidence.

## License

This project is licensed under the [MIT License](LICENSE).
