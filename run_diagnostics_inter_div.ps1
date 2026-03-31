$ErrorActionPreference = "Stop"

$model_name = "resnet1d"
$checkpoint = "checkpoints/ablation_meanrms_interdiv.pth"
$python = "D:\SoftWare\Anaconda\envs\WiFiPose\python.exe"

Write-Host "Waiting for training checkpoint to be produced..."
While (!(Test-Path $checkpoint)) {
    Start-Sleep -Seconds 10
}
Write-Host "Checkpoint found. Note: We should ideally wait until the training script is fully finished. We will check if python is still running."
# Check for the main train script in background
$isRunning = $true
while ($isRunning) {
    # It might be hard to distinguish the background train process from other python processes securely in powershell
    # So we simply wait until the logs stop updating... Wait, even better: check locks on checkpoint or wait for log "Training completed".
    $logs = Get-ChildItem -Path logs/train/ablation_meanrms_interdiv_resnet1d_*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($logs) {
        $lastLine = Get-Content $logs.FullName -Tail 1
        if ($lastLine -match "\[eval_end.*test_nMPJPE") {
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

Write-Host "All validation and diagnostic steps completed for the inter_div experiment."
