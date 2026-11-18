![BEE detector — Bee and hive segmentation dataset](banner.png)

**BEE detector** is a Linux dataset repository for bee and hive segmentation,
built to give beekeepers better resources to understand and support their hives.
Run `cargo run --release -- bee-data` to open the native Rust window: browse images
in a bento grid, inspect sources, collection dates and resolutions, and move source folders
or update their SQL locations. SQLite defines the train, val and test splits
independently of the folder layout. The attached SQL catalog holds only paths
and metadata; Git LFS versions the images and masks.
Labels cover workers, drones, queens, pollen, honey, pests and visible disease
signs, listed in [data/labels.txt](data/labels.txt).
[GVAT](https://github.com/jldx-1/gvat) is used to label and review the images and
segmentation annotations. The [guide](GUIDE.md) covers setup and PyTorch loading.
