param([switch]$Headless,[switch]$Demo,[switch]$Prepare)
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    $env:OMNI_KIT_ACCEPT_EULA='YES'
    $assetPython='../../work/venv/Scripts/python.exe'
    $isaacPython='../../work/isaac-venv/Scripts/python.exe'
    function Check-Report([string]$Path,[DateTime]$Started) {
        $file=Get-Item -LiteralPath $Path
        if ($file.LastWriteTimeUtc -lt $Started -or (Get-Content $file.FullName -Raw|ConvertFrom-Json).status -ne 'passed') {
            throw 'Isaac validation failed; inspect report.json'
        }
    }
    if ($Prepare -or -not (Test-Path 'out/ct_appendectomy/bimanual_scene/ct_appendectomy.usda')) {
        & $assetPython ct_appendectomy/prepare_cavity.py
        if ($LASTEXITCODE) {throw 'Asset preparation failed'}
        $started=[DateTime]::UtcNow
        & $isaacPython ct_appendectomy/run_scene.py --scene out/ct_appendectomy/cavity/ct_appendectomy.usda --bimanual --headless --frames 360 --out out/ct_appendectomy/bimanual_seed
        Check-Report 'out/ct_appendectomy/bimanual_seed/report.json' $started
        & $assetPython ct_appendectomy/plan_bimanual_camera.py --scene out/ct_appendectomy/cavity/ct_appendectomy.usda --run out/ct_appendectomy/bimanual_seed --out out/ct_appendectomy/bimanual_scene
        if ($LASTEXITCODE) {throw 'Camera planning failed'}
    }
    $runArgs=@('ct_appendectomy/run_scene.py','--scene','out/ct_appendectomy/bimanual_scene/ct_appendectomy.usda',
        '--bimanual','--frames','360','--out','out/ct_appendectomy/bimanual_desktop')
    if ($Headless) {$runArgs+='--headless'} else {$runArgs+='--interactive'}
    if ($Demo) {$runArgs+='--bimanual-demo'}
    $started=[DateTime]::UtcNow
    & $isaacPython @runArgs
    Check-Report 'out/ct_appendectomy/bimanual_desktop/report.json' $started
} finally {Pop-Location}
