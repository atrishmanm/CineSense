# Watches the Kaggle training kernel until it completes or fails.
# Run in background: powershell -File scripts/watch_kaggle.ps1
$slug = 'atrishmanmukherjee/cinesense-t4x2-deep-recommendation-training'
for ($i = 0; $i -lt 60; $i++) {
    $s = kaggle kernels status $slug
    Write-Output ("{0} {1}" -f (Get-Date -Format HH:mm:ss), $s)
    if ($s -match 'COMPLETE') { Write-Output 'KAGGLE_DONE'; break }
    if ($s -match 'FAILED|CANCELLED') { Write-Output 'KAGGLE_BAD'; break }
    Start-Sleep -Seconds 300
}
