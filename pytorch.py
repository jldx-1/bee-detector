"""Optional PyTorch reader. The Rust app owns the catalog; no Python backend."""

import argparse
import os
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


class BeeDataset(Dataset):
    """Snapshot SQL mappings once; decode pixels lazily in DataLoader workers.

    Returns three-channel RGB CHW images and uint8 HW masks.
    Default images stay uint8; high_range=True returns float32, preserving HDR.
    `size` is (height, width). Palette PNG masks retain their original class IDs.
    """

    def __init__(
        self,
        catalog="data/catalog.sqlite3",
        split="train",
        size=(512, 512),
        high_range=False,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("Expected train, val, or test")
        path = Path(catalog).resolve(strict=True)
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            self.records = [
                (str(path.parent / root / image), str(path.parent / root / mask))
                for root, image, mask in connection.execute(
                    "SELECT root_path, image_path, mask_path FROM dataset WHERE split=? ORDER BY id",
                    (split,),
                )
            ]
            self.classes = list(
                connection.execute(
                    "SELECT id, name FROM labels WHERE id != 255 ORDER BY id"
                )
            )
        finally:
            connection.close()
        if [label[0] for label in self.classes] != list(range(len(self.classes))):
            raise ValueError(
                "Training IDs must be contiguous from zero; 255 is ignored"
            )
        self.num_classes = len(self.classes)
        self.size = size
        self.high_range = high_range

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, mask_path = self.records[index]
        image = read_rgb(image_path, self.high_range)
        with Image.open(mask_path) as source:
            if source.mode not in ("L", "P"):
                raise ValueError("Mask must be grayscale or indexed PNG")
            mask = source.copy()
        if (image.shape[1], image.shape[0]) != mask.size:
            raise ValueError("Image and mask dimensions differ")
        if self.size:
            height, width = self.size
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
        return (
            torch.from_numpy(image.transpose(2, 0, 1).copy()),
            torch.from_numpy(np.array(mask, dtype=np.uint8, copy=True)),
        )


def read_rgb(path, high_range=False):
    """Decode without reducing precision; OpenCV's BGR output becomes RGB."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input images must have exactly three RGB channels")
    if image.dtype not in (np.uint8, np.uint16, np.float32, np.float64):
        raise ValueError(f"Unsupported RGB pixel type: {image.dtype}")
    image = image[:, :, ::-1]
    if not high_range:
        if image.dtype != np.uint8:
            raise ValueError(
                "Enable high_range=True or --high-range for 16-bit/HDR RGB"
            )
        return image
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / np.iinfo(image.dtype).max
    else:
        image = image.astype(np.float32)
    if not np.isfinite(image).all():
        raise ValueError("HDR RGB values must be finite and representable as float32")
    return image


def to_device(images, masks, device):
    """Scale byte images once; high-range float32 values are already prepared."""
    byte_images = images.dtype == torch.uint8
    images = images.to(device, dtype=torch.float32, non_blocking=True)
    if byte_images:
        images = images.div_(255)
    return images, masks.to(device, dtype=torch.long, non_blocking=True)


def make_loader(catalog, split="train", workers=4, high_range=False):
    dataset = BeeDataset(catalog, split, high_range=high_range)
    if not len(dataset):
        raise ValueError(
            f"The {split} split is empty; import annotated images in the Rust app"
        )
    return dataset, DataLoader(
        dataset,
        batch_size=8,
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        **({"prefetch_factor": 2} if workers > 0 else {}),
    )


def self_test():
    """Isolated fixtures verify SQL path resolution, palette IDs, and workers."""
    import pickle
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        collection = root / "arbitrary" / "nested"
        collection.mkdir(parents=True)
        Image.new("RGB", (31, 17), "gold").save(collection / "frame.jpg")
        ids = np.full((17, 31), 3, dtype=np.uint8)
        ids[:, 15:] = 255
        mask = Image.frombytes("P", (31, 17), ids.tobytes())
        mask.putpalette([value for n in range(256) for value in (255 - n, n, 100)])
        mask.save(collection / "frame.mask.png")
        rgb16 = np.full((17, 31, 3), (4096, 32768, 65535), dtype=np.uint16)
        hdr = np.full((17, 31, 3), (0.125, 1.5, 12.0), dtype=np.float32)
        assert cv2.imwrite(str(collection / "rgb16.png"), rgb16[:, :, ::-1])
        assert cv2.imwrite(
            str(collection / "hdr.tiff"),
            hdr[:, :, ::-1],
            [cv2.IMWRITE_TIFF_COMPRESSION, 1],
        )
        mask.save(collection / "rgb16.mask.png")
        mask.save(collection / "hdr.mask.png")
        db = root / "catalog.sqlite3"
        connection = sqlite3.connect(db)
        try:
            connection.executescript(
                Path(__file__).with_name("data").joinpath("catalog.sql").read_text()
            )
            connection.execute(
                "INSERT INTO sources(name,root_path,collected_at) VALUES (?,?,?)",
                ("hive-1__2026-06-12__camera-a", "arbitrary", "2026-06-12"),
            )
            connection.execute("INSERT INTO groups VALUES ('hive-1','train')")
            connection.execute("INSERT INTO groups VALUES ('hive-1-hdr','val')")
            connection.execute(
                "INSERT INTO samples(source,group_name,image_path,mask_path,image_sha256,mask_sha256,width,height) VALUES (?,?,?,?,?,?,?,?)",
                (
                    "hive-1__2026-06-12__camera-a",
                    "hive-1",
                    "nested/frame.jpg",
                    "nested/frame.mask.png",
                    "fixture",
                    "fixture",
                    31,
                    17,
                ),
            )
            for name, image_file in (("rgb16", "rgb16.png"), ("hdr", "hdr.tiff")):
                connection.execute(
                    "INSERT INTO samples(source,group_name,image_path,mask_path,image_sha256,mask_sha256,width,height) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        "hive-1__2026-06-12__camera-a",
                        "hive-1-hdr",
                        f"nested/{image_file}",
                        f"nested/{name}.mask.png",
                        name,
                        name,
                        31,
                        17,
                    ),
                )
            connection.commit()
        finally:
            connection.close()
        dataset = BeeDataset(db, size=(20, 28))
        dataset = pickle.loads(pickle.dumps(dataset))
        assert dataset.num_classes == 14
        for workers in (0, 2):
            images, masks = next(
                iter(DataLoader(dataset, num_workers=workers, batch_size=1))
            )
            assert images.shape == (1, 3, 20, 28)
            assert masks.shape == (1, 20, 28)
            assert images.dtype == masks.dtype == torch.uint8
            assert set(masks.unique().tolist()) == {3, 255}
            logits = torch.zeros((1, dataset.num_classes, 20, 28), requires_grad=True)
            loss = torch.nn.CrossEntropyLoss(ignore_index=255)(logits, masks.long())
            loss.backward()
            assert torch.isfinite(loss)
        high_range = BeeDataset(db, split="val", size=(20, 28), high_range=True)
        expected = torch.tensor((4096, 32768, 65535), dtype=torch.float32) / 65535
        assert torch.allclose(high_range[0][0][:, 0, 0], expected, atol=1e-7)
        assert torch.allclose(
            high_range[1][0][:, 0, 0], torch.tensor((0.125, 1.5, 12.0))
        )
        for workers in (0, 2):
            images, masks = next(
                iter(DataLoader(high_range, batch_size=2, num_workers=workers))
            )
            images, masks = to_device(images, masks, "cpu")
            assert images.shape == (2, 3, 20, 28)
            assert images.dtype == torch.float32 and images.max().item() == 12.0
            assert masks.dtype == torch.long and set(masks.unique().tolist()) == {
                3,
                255,
            }
        try:
            BeeDataset(db, split="val")[0]
        except ValueError as error:
            assert "high_range=True" in str(error)
        else:
            raise AssertionError(
                "High-range images must never be silently reduced to uint8"
            )
    print(
        "PyTorch reader: RGB channels, 16-bit/HDR values, palette masks, workers, and loss verified"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.sqlite3")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--high-range", action="store_true", help="Preserve 16-bit/HDR RGB as float32"
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    dataset, loader = make_loader(
        args.catalog, args.split, args.workers, args.high_range
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, masks = next(iter(loader))
    images, masks = to_device(images, masks, device)
    print(f"{len(dataset)} images · {dataset.num_classes} classes · {device}")
    print(f"Images {tuple(images.shape)}, masks {tuple(masks.shape)}")


if __name__ == "__main__":
    main()
