from __future__ import annotations

import argparse
import math
from collections import Counter
from typing import Iterable, Iterator, List, Sequence, Tuple

from pathlib import Path

import torch
from torch.utils.data import Sampler, DataLoader

from .aoa_dataset import AOASampleDataset, sample_to_env


class StratifiedBatchSampler(Sampler[List[int]]):
    """Batch sampler that encourages diversity over (action, env) within each batch.

    - Works on index level; requires a metadata sequence where
      meta[i] = (action_id: str, env_id: str).
    - Tries to ensure at least `min_actions` distinct actions and
      `min_envs` distinct envs per batch when possible.
    """

    def __init__(
        self,
        meta: Sequence[Tuple[str, str]],    # List of (Action, Env) for sample in dataset.
        batch_size: int = 64,
        min_actions: int = 2,               # minimum action diversity per batch
        min_envs: int = 2,                  # minimum env diversity per batch
        shuffle: bool = True,
        drop_last: bool = True,
        generator: torch.Generator | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.meta = meta
        self.batch_size = batch_size
        self.min_actions = min_actions
        self.min_envs = min_envs
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        self._num_samples = len(self.meta)

    # Generate initial shuffle sample indices.
    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            if self.generator is None:
                indices = torch.randperm(self._num_samples).tolist()    # shuffle indices
            else:
                indices = torch.randperm(self._num_samples, generator=self.generator).tolist()
        else:
            indices = list(range(self._num_samples))

        n = len(indices)
        bsz = self.batch_size
        i = 0
        while i < n:
            # Slice in size of batch_size;
            batch_idx = indices[i : i + bsz]
            if len(batch_idx) < bsz and self.drop_last:
                break

            # statis current batch with how many action and env. (Diversity)
            def stats(ids: Iterable[int]) -> Tuple[int, int]:
                acts = set(self.meta[j][0] for j in ids)
                envs = set(self.meta[j][1] for j in ids)
                return len(acts), len(envs)

            # Check batch diversity.
            num_act, num_env = stats(batch_idx)
            if (num_act < self.min_actions or num_env < self.min_envs) and len(batch_idx) >= 2:
                # Attempt a few swaps with later indices
                max_swaps = 4 * bsz
                swaps = 0
                j = i + bsz
                while swaps < max_swaps and j < n and (num_act < self.min_actions or num_env < self.min_envs):
                    # Try replacing one element in batch with indices[j]
                    replaced = False
                    for local_pos in range(len(batch_idx)):
                        candidate_batch = batch_idx.copy()
                        candidate_batch[local_pos] = indices[j]
                        c_act, c_env = stats(candidate_batch)
                        if c_act >= self.min_actions and c_env >= self.min_envs:
                            batch_idx = candidate_batch
                            replaced = True
                            num_act, num_env = c_act, c_env
                            swaps += 1
                            break
                    j += 1
                    if replaced is False:
                        swaps += 1

            # Generate batch index.
            yield batch_idx
            i += bsz

    def __len__(self) -> int:
        if self.drop_last:
            return self._num_samples // self.batch_size
        return math.ceil(self._num_samples / self.batch_size)


def _build_meta_from_dataset(ds: AOASampleDataset) -> List[Tuple[str, str]]:
        # AOASampleDataset.index entries are (action, sample, frame_idx)
        meta: List[Tuple[str, str]] = []
        for action, sample, _ in ds.index:
            env = sample_to_env(sample)     # S01-S10 -> env1, S11-S20 -> env2, ...
            meta.append((action, env))
        return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect stratified batch sampler over (action, env)")
    parser.add_argument("--aoa_root", type=str, default="data/aoa_cache")
    parser.add_argument("--labels_root", type=str, default="data/dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--drop_last", action="store_true")
    args = parser.parse_args()

    # project root is the parent of the "dataloader" package
    project_root = Path(__file__).resolve().parent.parent
    aoa_root = Path(args.aoa_root)
    labels_root = Path(args.labels_root)
    if not aoa_root.is_absolute():
        aoa_root = project_root / aoa_root
    if not labels_root.is_absolute():
        labels_root = project_root / labels_root

    ds = AOASampleDataset(aoa_root=aoa_root, labels_root=labels_root)
    meta = _build_meta_from_dataset(ds)

    sampler = StratifiedBatchSampler(
        meta=meta,
        batch_size=args.batch_size,
        min_actions=2,
        min_envs=2,
        shuffle=not args.no_shuffle,
        drop_last=args.drop_last,
    )

    dl = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    for b_idx, (_, _, batch_meta) in enumerate(dl):
        # batch_meta is a list of dicts
        actions = [m["action"] for m in batch_meta]
        envs = [m["env_id"] for m in batch_meta]
        c_act = Counter(actions)
        c_env = Counter(envs)
        print(f"Batch {b_idx}:")
        print("  actions:", dict(c_act))
        print("  envs   :", dict(c_env))
        if b_idx + 1 >= args.num_batches:
            break


if __name__ == "__main__":
    main()
