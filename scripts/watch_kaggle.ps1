# Watches the Kaggle training kernel until it completes or fails.
# Run in background: powershell -File scripts/watch_kaggle.ps1
#
# Exit codes:
#   0  kernel COMPLETE
#   1  kernel FAILED / CANCELLED / ERROR
#   2  watch timeout reached (5h) without a terminal state
#   127 kaggle CLI not found on PATH

$slug = 'atrishmanmukherjee/cinesense-t4x2-deep-recommendation-training'

# Fail fast if the kaggle CLI is unavailable.
if (-not (Get-Command kaggle -ErrorAction SilentlyContinue)) {
    Write-Output 'KAGGLE_CLI_MISSING'
    Write-Error 'kaggle CLI not found on PATH. Install with: pip install kaggle'
    exit 127
}

# 60 iterations x 300s = 5h cap.
$maxIterations = 60

for ($i = 0; $i -lt $maxIterations; $i++) {
    $s = kaggle kernels status $slug
    Write-Output ("{0} {1}" -f (Get-Date -Format HH:mm:ss), $s)
    if ($s -match 'COMPLETE') { Write-Output 'KAGGLE_DONE'; exit 0 }
    if ($s -match 'FAILED|CANCELLED|ERROR') { Write-Output 'KAGGLE_BAD'; exit 1 }
    Start-Sleep -Seconds 300
}

Write-Output 'KAGGLE_TIMEOUT'
exit 2
