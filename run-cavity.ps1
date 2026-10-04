param([switch]$Headless,[switch]$RecordScan)
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    & ../../work/venv/Scripts/python.exe ct_appendectomy/prepare_cavity.py
    if ($LASTEXITCODE) { throw 'Cavity asset preparation failed' }
    $env:OMNI_KIT_ACCEPT_EULA='YES'
    $runArgs=@('ct_appendectomy/run_scene.py','--scene','out/ct_appendectomy/cavity/ct_appendectomy.usda',
        '--explore-cavity','--frames','360','--out','out/ct_appendectomy/cavity_desktop')
    if ($Headless) {$runArgs+='--headless'} else {$runArgs+='--interactive'}
    if ($RecordScan) {$runArgs+='--record-scan'}
    $started=[DateTime]::UtcNow
    & ../../work/isaac-venv/Scripts/python.exe @runArgs
    $report=Get-Item 'out/ct_appendectomy/cavity_desktop/report.json'
    if ($report.LastWriteTimeUtc -lt $started -or (Get-Content $report.FullName -Raw | ConvertFrom-Json).status -ne 'passed') {
        throw 'Cavity validation failed; inspect report.json'
    }
} finally { Pop-Location }
