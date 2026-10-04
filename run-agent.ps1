param(
    [ValidateSet('scripted','random','checkpoint')][string]$Policy='scripted',
    [int]$Episodes=3,
    [int]$TrainSteps=0,
    [int]$Seed=42,
    [string]$OutputDirectory='out/ct_appendectomy/agent',
    [string]$Checkpoint,
    [switch]$Headless,
    [switch]$Prepare,
    [switch]$DemoWarmStart,
    [switch]$RgbObservation,
    [string]$IsaacPython=(Join-Path $PSScriptRoot '../../work/isaac-venv/Scripts/python.exe'),
    [string]$AssetPython=(Join-Path $PSScriptRoot '../../work/venv/Scripts/python.exe')
)
$ErrorActionPreference='Stop'
Push-Location $PSScriptRoot
try {
    $assets='out/ct_appendectomy/suite/forceps'
    if ($Prepare -or -not (Test-Path "$assets/forceps_config.json")) {
        & $AssetPython ct_appendectomy/build_scene.py --placement ct_appendectomy/profiles/s0042_hybrid.json
        if ($LASTEXITCODE) { throw 'Scene preparation failed' }
        $start=[DateTime]::UtcNow
        & $IsaacPython ct_appendectomy/run_scene.py --headless --forceps-probe --frames 480 --out $assets
        if ($LASTEXITCODE -or -not (Test-Path "$assets/report.json")) { throw 'Forceps preparation failed' }
        $probe=Get-Content "$assets/report.json" -Raw | ConvertFrom-Json
        if ($probe.status -ne 'passed' -or (Get-Item "$assets/report.json").LastWriteTimeUtc -lt $start) { throw 'Forceps preparation did not pass' }
    }
    $arguments=@('ct_appendectomy/agent_runner.py','--policy',$Policy,'--episodes',"$Episodes",'--train-steps',"$TrainSteps",'--seed',"$Seed",'--out',$OutputDirectory)
    if ($Headless) { $arguments+='--headless' }
    if ($Checkpoint) { $arguments+='--checkpoint';$arguments+=$Checkpoint }
    if ($DemoWarmStart) { $arguments+='--demo-warm-start' }
    if ($RgbObservation) { $arguments+='--rgb-observation' }
    $start=[DateTime]::UtcNow
    & $IsaacPython @arguments
    $agentExitCode=$LASTEXITCODE
    $path=Join-Path $OutputDirectory 'report.json'
    if (-not (Test-Path $path)) { throw 'No agent report' }
    $report=Get-Content $path -Raw | ConvertFrom-Json
    if ($agentExitCode -ne 0 -or (Get-Item $path).LastWriteTimeUtc -lt $start -or $report.status -eq 'failed' -or ($Episodes -gt 0 -and $report.status -ne 'passed')) {
        throw 'Agent run failed; inspect its report'
    }
} finally { Pop-Location }
