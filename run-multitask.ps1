param(
    [switch]$Prepare,
    [switch]$Train,
    [switch]$Headless,
    [ValidateSet('reach','dual_grasp','retract')][string]$Task='reach',
    [ValidateRange(1,10000000)][int]$Steps=2048
)
$ErrorActionPreference='Stop'
$simPython=Join-Path $PSScriptRoot '..\..\work\isaac-venv\Scripts\python.exe'
$simAssets=Join-Path $PSScriptRoot 'out\ct_appendectomy\multitask_assets'
$env:OMNI_KIT_ACCEPT_EULA='YES'
Push-Location $PSScriptRoot
try {
    if ($Prepare) {
        & $simPython ct_appendectomy/run_scene.py --scene out/ct_appendectomy/bimanual_scene/ct_appendectomy.usda --bimanual --headless --frames 360 --out $simAssets
        $prepareReport=Get-Content (Join-Path $simAssets 'report.json') -Raw | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0 -or $prepareReport.status -ne 'passed') { throw 'Physics asset preparation failed' }
    }
    $runArgs=@('ct_appendectomy/bimanual_benchmark.py','--assets',$simAssets,'--tasks',$Task,'--check-rgb')
    $runReport='out/ct_appendectomy/multitask_benchmark/report.json'
    if ($Headless) { $runArgs+='--headless' }
    if ($Train) {
        $runArgs+=@('--train-steps',"$Steps",'--policy','checkpoint','--out','out/ct_appendectomy/bimanual_ppo')
        $runReport='out/ct_appendectomy/bimanual_ppo/report.json'
    }
    & $simPython @runArgs
    if ($LASTEXITCODE -ne 0) { throw 'RL run failed; inspect report.json' }
    $result=Get-Content $runReport -Raw | ConvertFrom-Json
    if ($result.status -ne 'completed') { throw 'RL run did not complete; inspect report.json' }
    Write-Output ("Episodes completed; successes: {0}/{1}" -f $result.successes,$result.episodes.Count)
} finally { Pop-Location }
