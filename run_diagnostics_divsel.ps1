$ErrorActionPreference = "Stop"

$checkpoint = "checkpoints/ablation_meanrms_interdiv_divsel.pth"
$python = "D:\SoftWare\Anaconda\envs\WiFiPose\python.exe"

Write-Host "Waiting for training completion... (Monitoring log for '[eval_end.*test_nMPJPE')"
$isRunning = $true
while ($isRunning) {
    $logs = Get-ChildItem -Path logs/train/ablation_meanrms_interdiv_divsel_resnet1d_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($logs) {
        $lastLine = Get-Content $logs.FullName -Tail 1
        if ($lastLine -match "\[eval_end.*test_nMPJPE|\[assessment\] status") {
            $isRunning = $false
            Write-Host "Training finished!"
        } else {
            Start-Sleep -Seconds 15
        }
    } else {
        Start-Sleep -Seconds 15
    }
}

Write-Host "Step 1: Running eval.py..."
& $python eval.py --config configs/default.yaml --checkpoint $checkpoint --aoa_cache_root D:\Files\WiFi_Pose\WiFiPoseV3\data\AOA_data --labels_root D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset --normalize_mode mean_rms

Write-Host "Step 2: Running diagnose_pose_collapse.py..."
& $python diagnose_pose_collapse.py --config configs/default.yaml --checkpoint_glob $checkpoint --aoa_cache_root D:\Files\WiFi_Pose\WiFiPoseV3\data\AOA_data --labels_root D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset

Write-Host "Step 3: Running tools/diagnose_input_pose_separability.py..."
& $python tools/diagnose_input_pose_separability.py --aoa_cache_root D:\Files\WiFi_Pose\WiFiPoseV3\data\AOA_data --labels_root D:\Files\WiFi_Pose\WiFiPoseV3\data\dataset

Write-Host "All validation and diagnostic steps completed for the diversity_first experiment."