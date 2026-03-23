from pathlib import Path
from typing import List, Tuple, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def sample_to_env(sample_id: str) -> str:
    try:
        n = int(sample_id.lstrip('S').lstrip('s'))
    except Exception:
        return 'env0'
    env_idx = (n - 1) // 10 + 1
    return f'env{env_idx}'


class AOASampleDataset(Dataset):
    """Simple dataset that iterates all A01-A27 / S01-S40 samples.

    Each item is a single frame: (X: torch.Tensor (1,181), Y: torch.Tensor (17,2), meta: dict).
    """

    def __init__(self, aoa_root: str | Path, labels_root: str | Path, transform=None, window_size: int = 1):
        self.aoa_root = Path(aoa_root)
        self.labels_root = Path(labels_root)
        self.transform = transform
        self.window_size = max(1, int(window_size))
        if self.window_size % 2 == 0:
            self.window_size += 1
        self.window_radius = self.window_size // 2

        self.index: List[Tuple[str, str, int]] = []
        self._h5_map: Dict[Tuple[str, str], Path] = {}

        actions = [f"A{idx:02d}" for idx in range(1, 28)]
        samples = [f"S{idx:02d}" for idx in range(1, 41)]

        for action in actions:
            for sample in samples:
                h5_path = self.aoa_root / action / f"{sample}.h5"
                if not h5_path.exists():
                    # skip missing cache silently
                    continue
                with h5py.File(h5_path, 'r') as hf:
                    if 'aoa_spectrum' not in hf:
                        continue
                    num_frames = int(hf['aoa_spectrum'].shape[0])
                self._h5_map[(action, sample)] = h5_path
                for fi in range(num_frames):
                    self.index.append((action, sample, fi))

    @staticmethod
    def _normalize_aoa(aoa: np.ndarray) -> np.ndarray:
        aoa = np.asarray(aoa, dtype=np.float32)
        aoa = np.nan_to_num(aoa, nan=0.0, posinf=0.0, neginf=0.0)
        aoa = np.maximum(aoa, 0.0)
        aoa = np.log1p(aoa)
        p99 = float(np.percentile(aoa, 99.0))
        if p99 > 0:
            aoa = np.clip(aoa / p99, 0.0, 1.0)
        max_abs = float(np.max(np.abs(aoa)))
        if max_abs > 0:
            aoa = aoa / max_abs
        return aoa.astype(np.float32)

    @staticmethod
    def _normalize_pose(label: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose = np.asarray(label, dtype=np.float32).reshape(17, 2)
        center = pose.mean(axis=0, keepdims=True)
        centered = pose - center
        scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
        if scale < 1e-6:
            scale = 1.0
        normalized = centered / scale
        return normalized.astype(np.float32), center.astype(np.float32), np.array([scale], dtype=np.float32)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        action, sample, frame_idx = self.index[idx]
        h5_path = self._h5_map[(action, sample)]
        with h5py.File(h5_path, 'r') as hf:
            aoa_frames = hf['aoa_spectrum']
            num_frames = int(aoa_frames.shape[0])
            window = []
            for offset in range(-self.window_radius, self.window_radius + 1):
                source_idx = min(max(frame_idx + offset, 0), num_frames - 1)
                window.append(self._normalize_aoa(np.asarray(aoa_frames[source_idx], dtype=np.float32)))
            aoa = np.stack(window, axis=0)
            # try label_files in h5 if present
            label_path = None
            if 'label_files' in hf:
                raw = hf['label_files'][:]
                if isinstance(raw[0], (bytes, bytearray)):
                    label_path = Path(raw[frame_idx].decode())
                else:
                    label_path = Path(raw[frame_idx])

        # If h5 provides relative label paths (e.g., "A01/S01/rgb/frame001.npy"),
        # interpret them as relative to labels_root so the same cache works on
        # both Windows and Linux.
        if label_path is not None:
            label_str = str(label_path).replace("\\", "/")
            candidate = Path(label_str)
            if not candidate.is_absolute():
                candidate = self.labels_root / candidate
            if candidate.exists():
                label_path = candidate
            else:
                label_path = None

        if label_path is None:
            # default label location
            candidate = self.labels_root / action / sample / 'rgb'
            if not candidate.exists():
                candidate = self.labels_root / action / sample
            files = sorted(candidate.glob('*.npy'))
            label_path = files[frame_idx]

        label = np.load(label_path).astype(np.float32)

        X = aoa
        Y, pose_center, pose_scale = self._normalize_pose(label)

        if self.transform:
            X, Y = self.transform(X, Y)

        meta = {
            'env_id': sample_to_env(sample),
            'action': action,
            'sample': sample,
            'frame_idx': frame_idx,
            'window_size': self.window_size,
            'aoa_h5': str(h5_path),
            'label_path': str(label_path),
            'pose_center': pose_center,
            'pose_scale': pose_scale,
        }

        return torch.from_numpy(X).float(), torch.from_numpy(Y).float(), meta


if __name__ == '__main__':
    import argparse
    from pathlib import Path as _Path

    # project root is the parent directory of this "dataloader" package
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--aoa_root', required=True, default='data/aoa_cache')
    parser.add_argument('--labels_root', required=True, default='data/dataset')
    args = parser.parse_args()

    aoa_root = _Path(args.aoa_root)
    labels_root = _Path(args.labels_root)
    if not aoa_root.is_absolute():
        aoa_root = PROJECT_ROOT / aoa_root
    if not labels_root.is_absolute():
        labels_root = PROJECT_ROOT / labels_root

    ds = AOASampleDataset(aoa_root=aoa_root, labels_root=labels_root)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    x, y, meta = next(iter(dl))
    print(x.shape, y.shape, meta)
