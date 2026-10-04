<#
.SYNOPSIS
Run the restored Rosota reference task (rabbit anatomy, assisted grasp).
.DESCRIPTION
Uses the local Isaac runtime and original controller. This is a scripted
reference, not a learned policy or a CT baseline-parity claim. A first run
identifies the force model. Video playback follows simulation time.
#>
param(
    [switch]$Gui,
    [ValidateRange(1, 100)][int]$Episodes = 1,
    [ValidateRange(1, 300)][double]$Seconds = 25
)
$simRoot = $PSScriptRoot
$simPython = Join-Path $simRoot '../../work/isaac-venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $simPython)) { throw 'Local Isaac Python runtime is missing.' }
if (-not (Test-Path -LiteralPath (Join-Path $simRoot 'out/surgical_cell.usd'))) {
    throw 'Build the reference tissue and surgical cell with local_validation/run_rosota_reference.py first.'
}
$simOutput = 'out/reference_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
$simArgs = @('local_validation/run_rosota_reference.py', 'rollout', '--episodes', $Episodes,
    '--seconds', $Seconds, '--out', $simOutput, '--video', '--keep-frames', '--stereo', '--video-fps', '3.75')
if ($Gui) { $simArgs += '--gui' }
$simOldPath = $env:PATH
Push-Location -LiteralPath $simRoot
try {
    $env:PATH = (Join-Path $simRoot '.cache/bin') + ';' + $simOldPath
    & $simPython @simArgs
    if (-not (Test-Path -LiteralPath (Join-Path $simOutput 'summary.json'))) {
        throw 'No completed rollout report was written. Review the Isaac output.'
    }
    Write-Output ('Completed report: ' + (Join-Path $simRoot "$simOutput/summary.json"))
} finally {
    $env:PATH = $simOldPath
    Pop-Location
}
