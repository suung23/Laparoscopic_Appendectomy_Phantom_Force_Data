param([switch]$Visible,[ValidateRange(1,3000)][int]$Steps=90)
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    $env:OMNI_KIT_ACCEPT_EULA='YES'
    $isaacPython='../../work/isaac-venv/Scripts/python.exe'
    $assetPython='../../work/venv/Scripts/python.exe'
    $fr3='../../work/isaac-venv/Lib/site-packages/isaacsim/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/FR3/fr3.urdf'
    $output='out/ct_appendectomy/rosota_'+[Guid]::NewGuid().ToString('N')
    $runArgs=@('ct_appendectomy/run_rosota_ct.py','--fr3-urdf',$fr3,'--out',$output,'--steps',"$Steps")
    if (-not $Visible) {$runArgs+='--headless'}
    & $isaacPython @runArgs
    if ($LASTEXITCODE) {throw 'Isaac process failed'}
    $report=Get-Content -LiteralPath (Join-Path $output 'report.json') -Raw | ConvertFrom-Json
    if ($report.status -ne 'passed') {throw "Commissioning failed: $output/report.json"}
    & $assetPython ct_appendectomy/encode_exploration.py $output
    if ($LASTEXITCODE) {throw 'Video encoding failed'}
    Write-Output (Join-Path $PSScriptRoot "$output/exploration.mp4")
} finally {Pop-Location}
