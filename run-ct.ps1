param(
    [string]$AssetPython = (Join-Path $PSScriptRoot '../../work/venv/Scripts/python.exe'),
    [string]$IsaacPython = (Join-Path $PSScriptRoot '../../work/isaac-venv/Scripts/python.exe'),
    [switch]$Physics,
    [switch]$Contact,
    [switch]$Retraction,
    [switch]$RLSmoke,
    [switch]$Interactive
)
$ErrorActionPreference='Stop'
foreach ($python in @($AssetPython,$IsaacPython)) {
    if (-not (Test-Path -LiteralPath $python)) { throw 'Python environment missing; supply -AssetPython and -IsaacPython.' }
}
& $AssetPython (Join-Path $PSScriptRoot 'ct_appendectomy/build_scene.py')
if ($LASTEXITCODE) { throw 'CT asset build failed' }
$runArgs=@((Join-Path $PSScriptRoot 'ct_appendectomy/run_scene.py'))
if ($Physics) { $runArgs+='--physics' }
if ($Contact) { $runArgs+='--contact-probe' }
if ($Retraction) { $runArgs+='--retraction-probe' }
if ($RLSmoke) { $runArgs+='--rl-smoke' }
if ($Interactive) { $runArgs+='--interactive' } else { $runArgs+='--headless' }
$started=[DateTime]::UtcNow
& $IsaacPython @runArgs
$report=Join-Path $PSScriptRoot 'out/ct_appendectomy/isaac_probe/report.json'
if (-not (Test-Path -LiteralPath $report)) { throw 'Isaac produced no report; process exit alone is not a pass.' }
if ((Get-Item -LiteralPath $report).LastWriteTimeUtc -lt $started) { throw 'Isaac report is stale; this run failed.' }
$result=Get-Content -LiteralPath $report -Raw | ConvertFrom-Json
if ($result.status -ne 'passed') { throw 'Isaac probe failed; inspect its report.' }
Write-Host 'Isaac probe passed. See out/ct_appendectomy/isaac_probe.'
