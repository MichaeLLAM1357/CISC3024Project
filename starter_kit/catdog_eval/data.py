"""Data access: the raw LAION pool, the dev set, and crops described by a manifest.

Expected layout under the data root ($CATDOG_DATA, default ./data):

    data/student/index.csv                   pool_id, width, height
    data/student/laion_pool_student_NNN.zip  images/<pool_id>.jpg
    data/dev/dev.npz                         x [N,32,32,3] uint8, y [N] (0 = cat, 1 = dog)

The raw pool folder is data/student/ (as unpacked from student.zip); data/pool/ also works.

Images are returned as float tensors in [0, 1], shape [N, 3, 32, 32].
"""

import csv
import os
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

DEFAULT_DATA_ROOT = os.environ.get("CATDOG_DATA", "./data")
CLASSES = ("cat", "dog")
RESOLUTION = 32
# Accepted names of the raw pool folder under the data root, in order of preference.
POOL_DIRS = ("student", "pool")


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """uint8 [N, H, W, 3] -> float [N, 3, H, W] in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float().div_(255.0)


def load_image_set(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """(images, labels) from an .npz with x [N,32,32,3] uint8 and y [N]."""
    data = np.load(path)
    return to_tensor(data["x"]), torch.from_numpy(data["y"]).long()


def load_dev(root: str | Path = DEFAULT_DATA_ROOT) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(root) / "dev" / "dev.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; unpack the dev set there "
                                "(or set CATDOG_DATA)")
    return load_image_set(path)


class Pool:
    """Read-only access to the raw image pool (zip shards + index.csv)."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        index = self.folder / "index.csv"
        if not index.exists():
            raise FileNotFoundError(f"{index} not found; unpack the raw pool there")
        with open(index, newline="", encoding="utf-8") as f:
            self.index = {r["pool_id"]: r for r in csv.DictReader(f)}
        self._where: dict[str, Path] = {}
        self._open: dict[Path, zipfile.ZipFile] = {}
        for shard in sorted(self.folder.glob("*.zip")):
            with zipfile.ZipFile(shard) as z:
                for name in z.namelist():
                    self._where[Path(name).stem] = shard
        # Pools unpacked into a folder of images also work.
        for img in (self.folder / "images").glob("*.jpg"):
            self._where.setdefault(img.stem, img)

    def __contains__(self, pool_id: str) -> bool:
        return pool_id in self.index

    def size(self, pool_id: str) -> tuple[int, int]:
        r = self.index[pool_id]
        return int(r["width"]), int(r["height"])

    def image(self, pool_id: str) -> Image.Image:
        where = self._where.get(pool_id)
        if where is None:
            raise KeyError(f"{pool_id} is in index.csv but its image is missing")
        if where.suffix == ".zip":
            z = self._open.get(where) or self._open.setdefault(where, zipfile.ZipFile(where))
            data = z.read(f"images/{pool_id}.jpg")
            return Image.open(BytesIO(data)).convert("RGB")
        return Image.open(where).convert("RGB")


def pool_dir(root: str | Path = DEFAULT_DATA_ROOT) -> Path:
    """The raw pool folder under the data root: the first of data/student/, data/pool/ that
    has an index.csv (data/student/ if neither has)."""
    for name in POOL_DIRS:
        if (Path(root) / name / "index.csv").exists():
            return Path(root) / name
    return Path(root) / POOL_DIRS[0]


def default_pool(root: str | Path = DEFAULT_DATA_ROOT) -> Pool:
    return Pool(pool_dir(root))


def crop(image: Image.Image, x0: int, y0: int, size: int, angle: float = 0.0,
         res: int = RESOLUTION) -> Image.Image:
    """The manifest's crop: rotate the image by `angle` degrees counter-clockwise about the
    centre of the square (x0, y0, size), cut out the square, resize to res x res (Lanczos)."""
    if angle:
        image = image.rotate(angle, resample=Image.Resampling.BICUBIC,
                             center=(x0 + size / 2, y0 + size / 2))
    return image.crop((x0, y0, x0 + size, y0 + size)).resize(
        (res, res), Image.Resampling.LANCZOS)


def rebuild(rows: list[dict], pool: Pool) -> tuple[torch.Tensor, torch.Tensor]:
    """Images and labels described by manifest rows (see manifest.py)."""
    tiles, labels = [], []
    for r in rows:
        img = pool.image(r["pool_id"])
        tiles.append(np.asarray(crop(img, r["x0"], r["y0"], r["size"], r["angle"])))
        labels.append(r["label"])
    if not tiles:
        return torch.zeros(0, 3, RESOLUTION, RESOLUTION), torch.zeros(0, dtype=torch.long)
    return to_tensor(np.stack(tiles)), torch.tensor(labels, dtype=torch.long)
