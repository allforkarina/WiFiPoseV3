from pathlib import Path
from typing import List, Tuple, Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# Assumed COCO-style 17-joint order inferred from label geometry.
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_HIP = 11
RIGHT_HIP = 12


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

    def __init__(
        self,
        aoa_root: str | Path,
        labels_root: str | Path,
        transform=None,
        window_size: int = 1,
        normalize_mode: str = "pelvis_torso",
        input_mode: str = "diff",
        svd_rank: int = 1,
        feature_centering: bool = False,
        cache_in_memory: bool = False,
    ):
        self.aoa_root = Path(aoa_root)
        self.labels_root = Path(labels_root)
        self.transform = transform
        self.window_size = max(1, int(window_size))
        self.normalize_mode = str(normalize_mode).strip().lower() or "pelvis_torso"
        self.input_mode = str(input_mode).strip().lower() or "diff"
        self.svd_rank = max(1, int(svd_rank))
        self.feature_centering = bool(feature_centering)
        self.cache_in_memory = bool(cache_in_memory)
        if self.window_size % 2 == 0:
            self.window_size += 1
        self.window_radius = self.window_size // 2
        supported_input_modes = {"diff", "abs", "svd_residual", "svd_residual_diff"}
        if self.input_mode not in supported_input_modes:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}. Supported: {sorted(supported_input_modes)}")

        self.index: List[Tuple[str, str, int]] = []
        self._h5_map: Dict[Tuple[str, str], Path] = {}
        self._feature_cache: Dict[Tuple[str, str], np.ndarray] = {}

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

        if self.cache_in_memory:
            for action, sample in self._h5_map:
                self.load_feature_sequence(action, sample)

    @staticmethod
    def _normalize_aoa(aoa: np.ndarray, global_lower: float = -25.0, global_upper: float = 0.0) -> np.ndarray:
        aoa = np.asarray(aoa, dtype=np.float32)
        aoa = np.nan_to_num(aoa, nan=global_lower, posinf=global_upper, neginf=global_lower)
        
        # Phase 2 Step 2.1: Use fixed global physical thresholds for Min-Max scaling
        # rather than dynamic per-frame percentiles. This preserves the actual energy
        # level differences between moving and static frames (e.g., static energy is 
        # naturally much weaker, and this scaling retains that structural weakness).
        # We assume typical log-like bounds between -25.0 and 0.0.
        aoa = np.clip(aoa, global_lower, global_upper)
        aoa = (aoa - global_lower) / (global_upper - global_lower)
        return aoa.astype(np.float32)

    @staticmethod
    def _normalize_pose(label: np.ndarray, normalize_mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pose = np.asarray(label, dtype=np.float32).reshape(17, 2)
        mode = str(normalize_mode).strip().lower()
        if mode == "mean_rms":
            center = pose.mean(axis=0, keepdims=True)
            centered = pose - center
            scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
        elif mode == "pelvis_torso":
            pelvis = 0.5 * (pose[LEFT_HIP] + pose[RIGHT_HIP])
            shoulder_center = 0.5 * (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER])
            center = pelvis.reshape(1, 2)
            centered = pose - center
            torso_vec = shoulder_center - pelvis
            scale = float(np.linalg.norm(torso_vec))
        else:
            raise ValueError(f"Unsupported normalize_mode: {normalize_mode}")

        if scale < 1e-6:
            rms_scale = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
            scale = rms_scale if rms_scale >= 1e-6 else 1.0
        normalized = centered / scale
        return normalized.astype(np.float32), center.astype(np.float32), np.array([scale], dtype=np.float32)

    @staticmethod
    def _diff_sequence(sequence: np.ndarray) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.shape[0] == 0:
            return sequence.copy()
        prev_sequence = np.concatenate([sequence[:1], sequence[:-1]], axis=0)
        return (sequence - prev_sequence).astype(np.float32)

    @staticmethod
    def _compute_svd_residual(sequence: np.ndarray, rank: int) -> np.ndarray:
        sequence = np.asarray(sequence, dtype=np.float32)
        if sequence.ndim != 2:
            raise ValueError(f"SVD residual expects a 2D sequence, but got shape={sequence.shape}")
        if rank < 1:
            raise ValueError(f"svd_rank must be >= 1, but got {rank}")
        max_rank = min(sequence.shape)
        if rank >= max_rank:
            raise ValueError(
                f"svd_rank must be < min(num_frames, num_bins). Got rank={rank}, limit={max_rank} for shape={sequence.shape}"
            )
        u, s, vt = np.linalg.svd(sequence, full_matrices=False)
        static = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
        residual = sequence - static
        return residual.astype(np.float32)

    def _apply_feature_centering(self, sequence: np.ndarray) -> np.ndarray:
        if not self.feature_centering:
            return sequence.astype(np.float32)
        centered = sequence - sequence.mean(axis=0, keepdims=True)
        return centered.astype(np.float32)

    def _load_normalized_sequence(self, h5_path: Path, max_frames_per_sequence: int = 0) -> np.ndarray:
        with h5py.File(h5_path, 'r') as hf:
            aoa_frames = np.asarray(hf['aoa_spectrum'], dtype=np.float32)
        if max_frames_per_sequence > 0:
            aoa_frames = aoa_frames[:max_frames_per_sequence]
        return self._normalize_aoa(aoa_frames)

    def _build_feature_sequence(self, normalized_sequence: np.ndarray) -> np.ndarray:
        if self.input_mode == "abs":
            feature_sequence = normalized_sequence.copy()
        elif self.input_mode == "diff":
            feature_sequence = self._diff_sequence(normalized_sequence)
        elif self.input_mode == "svd_residual":
            feature_sequence = self._compute_svd_residual(normalized_sequence, self.svd_rank)
        elif self.input_mode == "svd_residual_diff":
            residual = self._compute_svd_residual(normalized_sequence, self.svd_rank)
            feature_sequence = self._diff_sequence(residual)
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")
        return self._apply_feature_centering(feature_sequence)

    def load_feature_sequence(
        self,
        action: str,
        sample: str,
        max_frames_per_sequence: int = 0,
    ) -> np.ndarray:
        seq_key = (action, sample)
        if max_frames_per_sequence <= 0 and seq_key in self._feature_cache:
            return self._feature_cache[seq_key]
        if seq_key not in self._h5_map:
            raise KeyError(f"Unknown sequence key: {seq_key}")
        h5_path = self._h5_map[seq_key]
        normalized_sequence = self._load_normalized_sequence(
            h5_path,
            max_frames_per_sequence=max_frames_per_sequence,
        )
        feature_sequence = self._build_feature_sequence(normalized_sequence)
        if max_frames_per_sequence <= 0 and self.cache_in_memory:
            self._feature_cache[seq_key] = feature_sequence
        return feature_sequence

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        action, sample, frame_idx = self.index[idx]
        h5_path = self._h5_map[(action, sample)]
        feature_sequence = self.load_feature_sequence(action, sample)
        num_frames = int(feature_sequence.shape[0])
        with h5py.File(h5_path, 'r') as hf:
            window = []
            for offset in range(-self.window_radius, self.window_radius + 1):
                source_idx = min(max(frame_idx + offset, 0), num_frames - 1)
                window.append(feature_sequence[source_idx])
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
        Y, pose_center, pose_scale = self._normalize_pose(label, self.normalize_mode)

        if self.transform:
            X, Y = self.transform(X, Y)

        meta = {
            'env_id': sample_to_env(sample),
            'action': action,
            'sample': sample,
            'frame_idx': frame_idx,
            'window_size': self.window_size,
            'input_mode': self.input_mode,
            'svd_rank': self.svd_rank,
            'feature_centering': self.feature_centering,
            'aoa_h5': str(h5_path),
            'label_path': str(label_path),
            'pose_center': pose_center,
            'pose_scale': pose_scale,
            'normalize_mode': self.normalize_mode,
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
