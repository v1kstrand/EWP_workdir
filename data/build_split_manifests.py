#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def discover_objects(dataset_root: Path) -> list[tuple[str, int]]:
    categories = sorted([path for path in dataset_root.iterdir() if path.is_dir()])
    label_map = {category.name: idx for idx, category in enumerate(categories)}
    objects: list[tuple[str, int]] = []
    for category in categories:
        label = label_map[category.name]
        for obj in sorted([path for path in category.iterdir() if path.is_dir()]):
            rel = "/" + str(obj.relative_to(dataset_root))
            objects.append((rel, label))
    return objects


def split_objects(
    objects: list[tuple[str, int]], train_split: float, seed: int
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    rng = random.Random(seed)
    shuffled = list(objects)
    rng.shuffle(shuffled)
    n_train = max(1, int(round(len(shuffled) * train_split)))
    if len(shuffled) > 1:
        n_train = min(n_train, len(shuffled) - 1)
    return shuffled[:n_train], shuffled[n_train:]


def save_split(split: list[tuple[str, int]], out_dir: Path, prefix: str) -> None:
    paths = np.asarray([path for path, _ in split])
    labels = np.asarray([label for _, label in split], dtype=np.int64)
    np.save(out_dir / f"{prefix}_images.npy", paths)
    np.save(out_dir / f"{prefix}_labels.npy", labels)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    objects = discover_objects(args.dataset_root)
    train_split, val_split = split_objects(objects, args.train_split, args.seed)

    save_split(train_split, args.out_dir, "train")
    save_split(val_split, args.out_dir, "val")

    print(f"dataset_root={args.dataset_root}")
    print(f"out_dir={args.out_dir}")
    print(f"seed={args.seed}")
    print(f"train_objects={len(train_split)}")
    print(f"val_objects={len(val_split)}")
    if train_split:
        print(f"train_example={train_split[0][0]}")
    if val_split:
        print(f"val_example={val_split[0][0]}")


if __name__ == "__main__":
    main()
