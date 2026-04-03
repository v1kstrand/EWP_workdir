#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rotations import euler_xyz_to_quat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--images-file", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-threads", type=int, default=16)
    return p.parse_args()


def _load_object_latents(dataset_root: Path, sample: str) -> tuple[np.ndarray, np.ndarray]:
    sample_root = dataset_root / sample[1:]
    latents = np.stack([
        np.load(sample_root / f"latent_{view_idx}.npy").astype(np.float32)
        for view_idx in range(50)
    ], axis=0)
    quats = euler_xyz_to_quat(torch.from_numpy(latents[:, :3])).cpu().numpy().astype(np.float32)
    return latents, quats


def _iter_results(dataset_root: Path, samples: np.ndarray, num_threads: int) -> Iterable[tuple[int, np.ndarray, np.ndarray]]:
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = {
            ex.submit(_load_object_latents, dataset_root, str(sample)): idx
            for idx, sample in enumerate(samples)
        }
        for fut in futures:
            idx = futures[fut]
            latents, quats = fut.result()
            yield idx, latents, quats


def main() -> None:
    args = parse_args()
    samples = np.load(args.images_file, allow_pickle=True)
    latents = np.empty((len(samples), 50, 7), dtype=np.float32)
    quats = np.empty((len(samples), 50, 4), dtype=np.float32)

    print(
        f"Building latent cache: samples={len(samples)} num_threads={args.num_threads} "
        f"output={args.output}"
    )
    for idx, sample_latents, sample_quats in _iter_results(args.dataset_root, samples, args.num_threads):
        latents[idx] = sample_latents
        quats[idx] = sample_quats
        if idx == 0 or (idx + 1) % 5000 == 0 or idx + 1 == len(samples):
            print(f"cached {idx + 1}/{len(samples)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, samples=samples, latents=latents, quats=quats)
    print(
        f"Wrote {args.output} | latents={latents.shape} {latents.dtype} | "
        f"quats={quats.shape} {quats.dtype}"
    )


if __name__ == "__main__":
    main()
