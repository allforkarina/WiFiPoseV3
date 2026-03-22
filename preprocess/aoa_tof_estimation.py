"""
Batch AoA (MUSIC) estimation and per-sample .h5 cache writer

How to use:
python preprocess/aoa_tof_estimation.py --raw_data_root data/raw --aoa_root data/aoa_cache --file_glob "*.mat" --require_frames 297
"""
import argparse
import pathlib
import sys
import time
from typing import Tuple, List

import numpy as np
import scipy.io  # for .mat loading
import h5py

# Reuse constants; adjust via CLI if needed
c = 3e8
NUM_ANTENNAS = 3
NUM_SUBCARRIERS = 114
D = 0.0258
F_CENTER = 5.8e9
DELTA_F = 80e6 / (NUM_SUBCARRIERS - 1)
THETA_GRID = np.arange(-90, 91, 1)
TAU_GRID_S = np.arange(0, 200, 2.0) * 1e-9
THRESH_RATIO = 0.08


# Cache for steering tables to avoid recomputing per frame
_STEERING_TABLE_CACHE: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}

def steering_aoa(theta_deg: float, num_antennas: int, d: float, wavelength: float) -> np.ndarray:
    m = np.arange(num_antennas)
    phase = -2j * np.pi * d * np.sin(np.deg2rad(theta_deg)) * m / wavelength
    return np.exp(phase)


def steering_tof(tau_s: float, num_subcarriers: int, f_center: float, delta_f: float) -> np.ndarray:
    k = np.arange(num_subcarriers)
    freqs = f_center + (k - (num_subcarriers - 1) / 2.0) * delta_f
    phase = -2j * np.pi * freqs * tau_s
    return np.exp(phase)


def build_smoothed_covariance(csi: np.ndarray, P: int, Q: int) -> np.ndarray:
    num_antennas, num_subcarriers = csi.shape
    num_sub_ant = num_antennas - P + 1
    num_sub_sc = num_subcarriers - Q + 1
    R = np.zeros((P * Q, P * Q), dtype=np.complex128)
    count = 0
    for ant_start in range(num_sub_ant):
        for sc_start in range(num_sub_sc):
            block = csi[ant_start:ant_start + P, sc_start:sc_start + Q]
            v = block.flatten(order="F")[:, None]
            R += v @ v.conj().T
            count += 1
    R /= max(count, 1)
    return R


def estimate_signal_subspaces(R: np.ndarray, thresh_ratio: float = 0.08):
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    maxeig = np.max(np.real(eigvals))
    num_signals = int(np.sum(np.real(eigvals) > thresh_ratio * maxeig))
    num_signals = max(num_signals, 1)
    En = eigvecs[:, num_signals:]
    return eigvals, num_signals, En


def steering_vec(theta_deg: float, tau_s: float, P: int, Q: int, d: float, wavelength: float, f_center: float, delta_f: float) -> np.ndarray:
    m = np.arange(P)
    phase_ant = -2j * np.pi * d * np.sin(np.deg2rad(theta_deg)) * m / wavelength
    a_ant = np.exp(phase_ant)
    k = np.arange(Q)
    freqs_block = f_center + (k - (Q - 1) / 2.0) * delta_f
    phase_tof = -2j * np.pi * freqs_block * tau_s
    a_tof = np.exp(phase_tof)
    a = np.kron(a_tof, a_ant)
    norm = np.linalg.norm(a)
    return a if norm == 0 else a / norm


def precompute_steering_tables(theta_grid: np.ndarray,
                               tau_grid_s: np.ndarray,
                               P: int,
                               Q: int,
                               d: float,
                               wavelength: float,
                               f_center: float,
                               delta_f: float) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute normalized steering vectors for AoA and ToF grids."""
    a_ant_table = np.empty((len(theta_grid), P), dtype=np.complex128)
    for i, theta in enumerate(theta_grid):
        m = np.arange(P)
        phase_ant = -2j * np.pi * d * np.sin(np.deg2rad(theta)) * m / wavelength
        vec = np.exp(phase_ant)
        norm = np.linalg.norm(vec)
        a_ant_table[i] = vec if norm == 0 else vec / norm

    k = np.arange(Q)
    freqs_block = f_center + (k - (Q - 1) / 2.0) * delta_f
    a_tof_table = np.empty((len(tau_grid_s), Q), dtype=np.complex128)
    for j, tau in enumerate(tau_grid_s):
        phase_tof = -2j * np.pi * freqs_block * tau
        vec = np.exp(phase_tof)
        norm = np.linalg.norm(vec)
        a_tof_table[j] = vec if norm == 0 else vec / norm

    return a_ant_table, a_tof_table


def get_steering_tables(P: int,
                        Q: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached steering tables (AoA and ToF) for given P, Q.

    These depend only on the fixed global grids and array parameters,
    so they can be safely reused across frames within the same process.
    """
    key = (P, Q)
    tables = _STEERING_TABLE_CACHE.get(key)
    if tables is not None:
        return tables

    wavelength = c / F_CENTER
    tables = precompute_steering_tables(
        THETA_GRID,
        TAU_GRID_S,
        P,
        Q,
        D,
        wavelength,
        F_CENTER,
        DELTA_F,
    )
    _STEERING_TABLE_CACHE[key] = tables
    return tables


def music_spectrum_2d(Pi: np.ndarray,
                      theta_grid: np.ndarray,
                      tau_grid_s: np.ndarray,
                      P: int,
                      Q: int,
                      d: float,
                      wavelength: float,
                      f_center: float,
                      delta_f: float,
                      a_ant_table: np.ndarray,
                      a_tof_table: np.ndarray) -> np.ndarray:
    spectrum = np.zeros((len(theta_grid), len(tau_grid_s)), dtype=np.float64)
    for i in range(len(theta_grid)):
        for j in range(len(tau_grid_s)):
            a = np.kron(a_tof_table[j], a_ant_table[i])
            denom = np.real(np.conj(a).T @ Pi @ a)
            denom = denom if denom > 1e-12 else 1e-12
            spectrum[i, j] = 1.0 / denom
    spectrum_db = 10 * np.log10(spectrum / np.max(spectrum))
    return spectrum_db, spectrum


def load_csi(
    path: pathlib.Path,
    snapshot_index: int | None = None,
    snapshot_agg: str | None = None,
    return_all_snapshots: bool = False,
) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as z:
            data = z[list(z.keys())[0]]
    elif suffix == ".mat":
        mat = scipy.io.loadmat(path)
        if "CSIamp" in mat:
            amp = np.asarray(mat["CSIamp"])
            phase = np.asarray(mat.get("CSIphase", np.zeros_like(amp)))
            if amp.shape != phase.shape:
                raise ValueError("CSIamp and CSIphase shape mismatch")
            if "CSIphase" not in mat:
                print("Warning: CSIphase missing, using zero phase (results will be inaccurate)", file=sys.stderr)
            data = amp * np.exp(1j * phase)
        else:
            raise ValueError("Expected CSIamp (and optionally CSIphase) in .mat file")
    else:
        # Expect CSV with real and imag interleaved as columns: a0r,a0i,a1r,a1i,... per subcarrier row
        data = np.loadtxt(path, delimiter=",")
    data = np.asarray(data)
    if np.iscomplexobj(data):
        csi = data
    else:
        if data.ndim == 2 and data.shape[1] == NUM_ANTENNAS * 2:
            real = data[:, 0::2]
            imag = data[:, 1::2]
            csi = real + 1j * imag
        else:
            raise ValueError("Expected complex data or CSV with real/imag interleaved columns per subcarrier")
    # Handle snapshots dimension if present (e.g., shape (ant, sc, snap))
    if csi.ndim == 3:
        if return_all_snapshots:
            if csi.shape[0] != NUM_ANTENNAS or csi.shape[1] != NUM_SUBCARRIERS:
                raise ValueError(f"CSI shape mismatch, got {csi.shape}, expected ({NUM_ANTENNAS}, {NUM_SUBCARRIERS}, Nsnap)")
            max_abs = np.max(np.abs(csi))
            if max_abs > 0:
                csi = csi / max_abs
            return csi
        if snapshot_index is not None:
            csi = csi[:, :, snapshot_index]
        elif snapshot_agg is not None:
            if snapshot_agg == "mean":
                csi = np.mean(csi, axis=2)
            elif snapshot_agg == "median":
                csi = np.median(csi, axis=2)
            else:
                raise ValueError("snapshot_agg must be None, 'mean', or 'median'")
        else:
            # default: use first snapshot
            csi = csi[:, :, 0]

    csi = csi.reshape(NUM_SUBCARRIERS, NUM_ANTENNAS).T if csi.shape != (NUM_ANTENNAS, NUM_SUBCARRIERS) else csi
    max_abs = np.max(np.abs(csi))
    if max_abs > 0:
        csi = csi / max_abs
    if csi.shape != (NUM_ANTENNAS, NUM_SUBCARRIERS):
        raise ValueError(f"CSI shape mismatch, got {csi.shape}, expected ({NUM_ANTENNAS}, {NUM_SUBCARRIERS})")
    return csi


def run(path: pathlib.Path,
        P: int,
        Q: int,
        thresh_ratio: float,
        snapshot_index: int | None,
        snapshot_agg: str | None,
        all_snapshots: bool,
        verbose: bool = False,
    ):
    wavelength = c / F_CENTER
    theta_grid = THETA_GRID
    tau_grid_s = TAU_GRID_S
    a_ant_table, a_tof_table = get_steering_tables(P, Q)
    csi = load_csi(
        path,
        snapshot_index=snapshot_index,
        snapshot_agg=snapshot_agg,
        return_all_snapshots=all_snapshots,
    )

    if all_snapshots:
        if csi.ndim != 3:
            raise ValueError("all_snapshots=True but CSI is not 3D")
        n_snap = csi.shape[2]
        spectra_db_list = []
        spectra_lin_accum = None
        for s in range(n_snap):
            csi_s = csi[:, :, s]
            R_smooth = build_smoothed_covariance(csi_s, P, Q)
            eigvals, num_signals, En = estimate_signal_subspaces(R_smooth, thresh_ratio=thresh_ratio)
            Pi = En @ En.conj().T
            spectrum_db, spectrum_lin = music_spectrum_2d(Pi, theta_grid, tau_grid_s, P, Q, D, wavelength, F_CENTER, DELTA_F, a_ant_table, a_tof_table)
            if spectra_lin_accum is None:
                spectra_lin_accum = np.zeros_like(spectrum_lin)
            spectra_lin_accum += spectrum_lin
            spectra_db_list.append(spectrum_db)

            if verbose:
                print(f"Snapshot {s}: CSI shape {csi_s.shape}, R {R_smooth.shape}, signals {num_signals}")
                flat_indices = np.argpartition(spectrum_db.flatten(), -1 * num_signals)[-1 * num_signals:]
                sorted_indices = flat_indices[np.argsort(spectrum_db.flatten()[flat_indices])[::-1]]
                for idx in sorted_indices:
                    i, j = np.unravel_index(idx, spectrum_db.shape)
                    print(f"  Peak: {spectrum_db[i, j]:.2f} dB at AoA={theta_grid[i]} deg, ToF={tau_grid_s[j]*1e9:.1f} ns")

        avg_lin = spectra_lin_accum / n_snap
        spectrum_db_avg = 10 * np.log10(avg_lin / np.max(avg_lin))
        return theta_grid, tau_grid_s, spectrum_db_avg, spectra_db_list, avg_lin

    R_smooth = build_smoothed_covariance(csi, P, Q)
    eigvals, num_signals, En = estimate_signal_subspaces(R_smooth, thresh_ratio=thresh_ratio)
    Pi = En @ En.conj().T
    spectrum_db, spectrum_lin = music_spectrum_2d(Pi, theta_grid, tau_grid_s, P, Q, D, wavelength, F_CENTER, DELTA_F, a_ant_table, a_tof_table)
    if verbose:
        print(f"CSI shape: {csi.shape}")
        print(f"Smoothed covariance shape: {R_smooth.shape}")
        print(f"Top eigenvalues: {np.real(eigvals[:min(5, len(eigvals))])}")
        print(f"Estimated signal count: {num_signals}")

        flat_indices = np.argpartition(spectrum_db.flatten(), -1 * num_signals)[-1 * num_signals:]
        sorted_indices = flat_indices[np.argsort(spectrum_db.flatten()[flat_indices])[::-1]]
        for idx in sorted_indices:
            i, j = np.unravel_index(idx, spectrum_db.shape)
            print(f"Peak: {spectrum_db[i, j]:.2f} dB at AoA={theta_grid[i]} deg, ToF={tau_grid_s[j]*1e9:.1f} ns")

    return theta_grid, tau_grid_s, spectrum_db, None, spectrum_lin


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch AOA (MUSIC) estimation and per-sample .h5 cache writer")
    parser.add_argument("--raw_data_root", type=pathlib.Path, default=pathlib.Path("data/dataset"),
                        help="Root directory of raw CSI, structure: <raw_data_root>/Axx/Syy/wifi-csi/*.mat (default: data/dataset)")
    parser.add_argument("--aoa_root", type=pathlib.Path, default=pathlib.Path("data/aoa_cache"),
                        help="Output root for per-sample .h5 files, structure: <aoa_root>/Axx/Syy.h5 (default: data/aoa_cache)")
    parser.add_argument("--labels_root", type=pathlib.Path, default=None,
                        help="Optional labels root; if unset, labels are assumed under raw_data_root/Axx/Syy/rgb")
    parser.add_argument("--file_glob", type=str, default="*.mat",
                        help="Glob pattern to find CSI frame files under each sample directory (default: '*.mat')")
    parser.add_argument("--label_glob", type=str, default="*.npy",
                        help="Glob pattern to find label frame files under each sample directory (default: '*.npy')")
    parser.add_argument("--P", type=int, default=2, help="Antennas per subarray")
    parser.add_argument("--Q", type=int, default=57, help="Subcarriers per sub-block")
    parser.add_argument("--snapshot-index", type=int, default=None,
                        help="Select snapshot index from 3D CSI; default uses first (or aggregated if specified)")
    parser.add_argument("--snapshot-agg", choices=["mean", "median"], default=None,
                        help="Aggregate snapshots if not using all")
    parser.add_argument("--all-snapshots", action="store_true",
                        help="Run estimation on all snapshots and average before ToF reduction")
    parser.add_argument("--require_frames", type=int, default=297,
                        help="Expected frames per sample; used for consistency checks")
    parser.add_argument("--on_mismatch", choices=["skip", "pad", "error"], default="error",
                        help="Behavior when a sample has frame count != require_frames or frames mismatch (default: error)")
    parser.add_argument("--pad_method", choices=["repeat", "zeros"], default="repeat",
                        help="If --on_mismatch pad, how to pad frames")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .h5 outputs")
    parser.add_argument("--preprocess_version", type=str, default="v1",
                        help="Preprocess version string to write into h5 metadata")
    parser.add_argument("--no_tof", action="store_true", default=True,
                        help="Ignore TOF in outputs (TOF estimation still runs internally if needed) - default True")
    parser.add_argument("--dry_run", action="store_true", help="List samples to process without running MUSIC")
    parser.add_argument("--bad_samples_log", type=pathlib.Path, default=pathlib.Path("data/bad_samples.log"),
                        help="File to append bad sample records")
    parser.add_argument("--check_complete", action="store_true",
                        help="When skipping existing .h5, verify aoa_spectrum shape == (require_frames, 181); "
                             "otherwise recompute")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args(argv)

    # Resolve paths relative to project root (parent of "preprocess" directory)
    def resolve_root(p: pathlib.Path) -> pathlib.Path:
        if p is None:
            return None
        p = pathlib.Path(p)
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / p).resolve()

    raw_root: pathlib.Path = resolve_root(args.raw_data_root)
    aoa_root: pathlib.Path = resolve_root(args.aoa_root)
    labels_root: pathlib.Path = resolve_root(args.labels_root) if args.labels_root is not None else raw_root
    pattern = args.file_glob
    label_pattern = args.label_glob

    # find action/sample directories under raw_root
    action_dirs = sorted([p for p in raw_root.iterdir() if p.is_dir() and p.name.upper().startswith('A')])
    if not action_dirs:
        print(f"No action directories found under {raw_root}", file=sys.stderr)
        sys.exit(1)

    # helper to collect CSI and label frame files under required subfolders
    def collect_frames(sample_dir: pathlib.Path) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
        csi_dir = sample_dir / 'wifi-csi'
        rgb_dir = sample_dir / 'rgb'
        # CSI files
        if csi_dir.exists() and csi_dir.is_dir():
            csi_files = sorted(csi_dir.glob(pattern))
        else:
            # backward compatibility: accept direct files under sample_dir
            csi_files = sorted(sample_dir.glob(pattern))
        # label files
        if rgb_dir.exists() and rgb_dir.is_dir():
            label_files = sorted(rgb_dir.glob(label_pattern))
        else:
            label_files = sorted(sample_dir.glob(label_pattern))
        return csi_files, label_files

    aoa_root.mkdir(parents=True, exist_ok=True)

    # Build full sample list for progress reporting
    sample_dirs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for action_dir in action_dirs:
        for sample_dir in sorted([p for p in action_dir.iterdir() if p.is_dir() and p.name.upper().startswith('S')]):
            sample_dirs.append((action_dir, sample_dir))
    total_samples = len(sample_dirs)

    # Detect already existing outputs for optional resume/skip behavior
    completed_samples: set[tuple[str, str]] = set()
    for action_dir, sample_dir in sample_dirs:
        action = action_dir.name
        sample = sample_dir.name
        out_h5 = aoa_root / action / f"{sample}.h5"
        if not out_h5.exists():
            continue
        if not args.check_complete:
            completed_samples.add((action, sample))
            continue
        # Strict completeness check: h5 must contain aoa_spectrum with expected shape
        try:
            with h5py.File(out_h5, 'r') as hf:
                if 'aoa_spectrum' not in hf:
                    continue
                dset = hf['aoa_spectrum']
                if dset.shape == (args.require_frames, 181):
                    completed_samples.add((action, sample))
        except OSError:
            # Corrupted or unreadable file; treat as incomplete
            continue

    existing = len(completed_samples)
    pending = total_samples - existing
    print(
        f"Total samples: {total_samples}, existing outputs: {existing}, "
        f"to process this run: {pending} (overwrite={'on' if args.overwrite else 'off'})"
    )

    # Record frames and process time
    total_processed = 0
    total_skipped = 0
    start_time = time.time()

    for action_dir, sample_dir in sample_dirs:
        action = action_dir.name            # Axx
        sample = sample_dir.name            # Syy
        out_dir = aoa_root / action     # AoA store in the same dir structure like Axx/Syy.h5
        out_dir.mkdir(parents=True, exist_ok=True)
        out_h5 = out_dir / f"{sample}.h5"
        sample_start = time.time()
        # Whether to skip or to overwrite (resume support)
        if not args.overwrite and (action, sample) in completed_samples:
            print(f"SKIP {action}/{sample} (exists; use --overwrite to recompute)")
            total_skipped += 1
            continue

        csi_files, label_files = collect_frames(sample_dir)
        if not csi_files:
            print(f"No CSI .mat files found for {action}/{sample} under {sample_dir}/wifi-csi, skipping", file=sys.stderr)
            total_skipped += 1
            continue
        if not label_files:
            print(f"No label .npy files found for {action}/{sample} under {sample_dir}/rgb, skipping", file=sys.stderr)
            total_skipped += 1
            continue

        if args.dry_run:
            print(f"Would process {action}/{sample}: CSI frames={len(csi_files)}, label frames={len(label_files)}")
            continue

        ordered_stems = [p.stem for p in csi_files]
        ordered_csi = csi_files
        ordered_labels = label_files

        aoa_list = []
        bad = False
        for idx, csi_path in enumerate(ordered_csi):
            label_path = ordered_labels[idx] if idx < len(ordered_labels) else None

            if csi_path is None:
                # padded CSI frame
                if args.on_mismatch == 'pad':
                    if not aoa_list:
                        print(f"Cannot pad first frame for {action}/{sample}; skipping sample", file=sys.stderr)
                        bad = True
                        break
                    if args.pad_method == 'repeat':
                        aoa_list.append(aoa_list[-1].copy())
                    else:
                        aoa_list.append(np.zeros((len(THETA_GRID),), dtype=np.float32))
                    continue
                else:
                    print(f"Missing CSI for stem {ordered_stems[idx]} in {action}/{sample}", file=sys.stderr)
                    bad = True
                    break

            try:
                theta_grid, tau_grid_s, spectrum_db, spectra_db_list, spectrum_lin = run(
                    path=csi_path,
                    P=args.P,
                    Q=args.Q,
                    thresh_ratio=THRESH_RATIO,
                    snapshot_index=args.snapshot_index,
                    snapshot_agg=args.snapshot_agg,
                    all_snapshots=args.all_snapshots,
                    verbose=args.verbose,
                )
            except Exception as e:
                print(f"Error processing {csi_path}: {e}", file=sys.stderr)
                bad = True
                break

            if spectrum_lin is None:
                print(f"No linear spectrum for {csi_path}, skipping sample {action}/{sample}", file=sys.stderr)
                bad = True
                break

            # average over ToF axis to get 1D AoA linear power
            aoa_lin = np.mean(spectrum_lin, axis=1)
            # ensure length matches theta_grid
            if aoa_lin.shape[0] != len(theta_grid):
                print(f"Unexpected AoA length {aoa_lin.shape[0]} for {csi_path}, expected {len(theta_grid)}", file=sys.stderr)
                bad = True
                break
            aoa_list.append(aoa_lin.astype(np.float32))

            # basic label existence/validity check (kept lightweight; no per-frame np.load)
            if label_path is None and args.on_mismatch == 'error':
                print(f"Missing label for frame {ordered_stems[idx]} in {action}/{sample}", file=sys.stderr)
                bad = True
                break

        if bad:
            # record and skip according to policy
            msg = f"Bad sample {action}/{sample}: error during processing"
            print(msg, file=sys.stderr)
            with open(args.bad_samples_log, 'a', encoding='utf8') as bf:
                bf.write(msg + '\n')
            total_skipped += 1
            continue

        num_frames = len(aoa_list)
        # consistency check
        if num_frames != args.require_frames:
            if args.on_mismatch == 'error':
                raise RuntimeError(f"Sample {action}/{sample} has {num_frames} frames, expected {args.require_frames}")
            elif args.on_mismatch == 'skip':
                print(f"Sample {action}/{sample} frame count {num_frames} != {args.require_frames}, skipping", file=sys.stderr)
                with open(args.bad_samples_log, 'a', encoding='utf8') as bf:
                    bf.write(f"Mismatch_frames {action}/{sample}: {num_frames}\n")
                total_skipped += 1
                continue
            elif args.on_mismatch == 'pad':
                if num_frames == 0:
                    print(f"No frames for {action}/{sample}, cannot pad", file=sys.stderr)
                    total_skipped += 1
                    continue
                if args.pad_method == 'repeat':
                    last = aoa_list[-1]
                    while len(aoa_list) < args.require_frames:
                        aoa_list.append(last.copy())
                else:
                    zeros = np.zeros_like(aoa_list[0])
                    while len(aoa_list) < args.require_frames:
                        aoa_list.append(zeros)
                num_frames = len(aoa_list)

        aoa_array = np.stack(aoa_list, axis=0)  # (num_frames, len(theta_grid))

        # final dimension check: aoa_resolution
        aoa_resolution = aoa_array.shape[1]
        if aoa_resolution != 181:
            # if theta grid differs, we will still save but warn
            print(f"Warning: AoA resolution {aoa_resolution} != 181 for {action}/{sample}", file=sys.stderr)

        # write to h5 atomically
        tmp_h5 = out_h5.with_suffix('.h5.tmp')
        with h5py.File(tmp_h5, 'w') as hf:
            dset = hf.create_dataset('aoa_spectrum', data=aoa_array, compression='gzip')
            hf.attrs['action'] = action
            hf.attrs['sample'] = sample
            hf.attrs['num_frames'] = num_frames
            hf.attrs['aoa_resolution'] = aoa_resolution
            hf.attrs['preprocess_version'] = args.preprocess_version
            hf.attrs['generated_at'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            # store source CSI and label file lists for traceability
            src_list = [str(p.relative_to(raw_root)) if p is not None else b'' for p in ordered_csi]
            lbl_list = [str(p.relative_to(raw_root)) if p is not None else b'' for p in ordered_labels]
            hf.create_dataset('source_files', data=np.array(src_list, dtype='S'))
            hf.create_dataset('label_files', data=np.array(lbl_list, dtype='S'))

        # move tmp to final
        tmp_h5.replace(out_h5)
        sample_elapsed = time.time() - sample_start
        total_processed += 1
        print(
            f"Processed {action}/{sample} in {sample_elapsed:.2f}s "
            f"({total_processed}/{total_samples}) -> {out_h5} (frames={num_frames}, res={aoa_resolution})"
        )

    elapsed = time.time() - start_time
    print(f"Done. Processed: {total_processed}, Skipped: {total_skipped}, time: {elapsed:.1f}s")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
