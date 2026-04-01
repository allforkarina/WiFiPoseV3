import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Using the COCO 17-keypoint order loosely referenced in the repository
# Wrist is typically 10 (Right) or 9 (Left) in standard COCO. Let's use 10.
RIGHT_WRIST = 10

def check_alignment(aoa_path: Path, label_dir: Path, output_path: Path):
    if not aoa_path.exists():
        print(f"Error: AOA file not found: {aoa_path}")
        return
    if not label_dir.exists():
        print(f"Error: Label dir not found: {label_dir}")
        return

    # Load AOA
    with h5py.File(aoa_path, 'r') as hf:
        aoa_data = hf['aoa_spectrum'][:]  # Shape typically (num_frames, 11, 181) or similar
        # Calculate mean energy per frame
        # If AOA is log-scale [-25, 0], we can just average the log values to represent gross power changes
        aoa_energy = np.mean(aoa_data, axis=1)
        
    num_frames = aoa_energy.shape[0]

    # Load Pos labels
    label_files = sorted(list(label_dir.glob('*.npy')))
    if len(label_files) == 0:
        print(f"Error: No labels found in {label_dir}")
        return
        
    if len(label_files) != num_frames:
        print(f"Warning: Frame count mismatch! AOA has {num_frames} frames, but found {len(label_files)} labels.")
        # Proceed with the smaller count to avoid crashes
        num_frames = min(num_frames, len(label_files))

    wrist_y_positions = []
    for i in range(num_frames):
        pos_data = np.load(label_files[i]).reshape(17, 2)
        wrist_y_positions.append(pos_data[RIGHT_WRIST, 1])
        
    wrist_y = np.array(wrist_y_positions)
    
    # Trim AOA to match labels if necessary
    aoa_energy = aoa_energy[:num_frames]

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:red'
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('AOA Mean Energy', color=color)
    ax1.plot(range(num_frames), aoa_energy, color=color, label='AOA Energy')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Right Wrist Y-Pos', color=color)  
    ax2.plot(range(num_frames), wrist_y, color=color, label='Right Wrist Y-Pos', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    plt.title(f"Data Alignment Sanity Check (AOA Energy vs Right Wrist Y_Pos)\n{aoa_path.parent.name}/{aoa_path.stem}")
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saved temporal alignment plot to {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aoa_file', type=str, default='data/AOA_data/A01/S01.h5')
    parser.add_argument('--label_dir', type=str, default='data/dataset/A01/S01/rgb')
    parser.add_argument('--output', type=str, default='logs/eval/data_sanity_check_A01_S01.png')
    args = parser.parse_args()
    
    # We will override with correct D: drive paths to fit the workstation config
    aoa = Path(r"D:\Files\WiFi_Pose\WiFiPoseV3\data\AOA_data\A01\S01.h5")
    lab = Path(r"D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset\A01\S01\rgb")
    out = Path(args.output)
    
    check_alignment(aoa, lab, out)