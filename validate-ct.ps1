param(
    [string]$AssetPython = (Join-Path $PSScriptRoot '../../work/venv/Scripts/python.exe'),
    [string]$IsaacPython = (Join-Path $PSScriptRoot '../../work/isaac-venv/Scripts/python.exe')
)
$ErrorActionPreference='Stop'
$results=@()
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path '.cache','out/ct_appendectomy/suite' | Out-Null
    @{status='starting';appendectomy_completed=$false} | ConvertTo-Json |
        Set-Content -Encoding UTF8 'out/ct_appendectomy/suite/report.json'
    & $AssetPython ct_appendectomy/build_scene.py --placement ct_appendectomy/profiles/s0042_hybrid.json
    if ($LASTEXITCODE) { throw 'Scene build failed' }
    $cases=@(
        @{name='render'; flags=@('--frames','8')},
        @{name='forceps'; flags=@('--forceps-rl-smoke','--frames','480')},
        @{name='forceps-control'; flags=@('--forceps-probe','--disable-tool-contact','--frames','480')},
        @{name='retraction'; flags=@('--rl-smoke','--frames','120')}
    )
    foreach ($case in $cases) {
        $directory=Join-Path 'out/ct_appendectomy/suite' $case.name
        $arguments=@('ct_appendectomy/run_scene.py','--headless','--require-appendix-visible','--out',$directory)+$case.flags
        $start=[DateTime]::UtcNow
        $timer=[Diagnostics.Stopwatch]::StartNew()
        & $IsaacPython @arguments *> (Join-Path '.cache' ('ct-suite-'+$case.name+'.log'))
        $exitCode=$LASTEXITCODE
        $timer.Stop()
        $path=Join-Path $directory 'report.json'
        if (-not (Test-Path -LiteralPath $path)) { throw "No report for $($case.name)" }
        if ((Get-Item -LiteralPath $path).LastWriteTimeUtc -lt $start) { throw "Stale report for $($case.name)" }
        $report=Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
        if ($exitCode -ne 0 -or $report.status -ne 'passed') { throw "Failed $($case.name); inspect $path" }
        if ($case.name -eq 'forceps' -and $report.forceps_rl_validation.status -ne 'passed') { throw 'Forceps RL checks missing' }
        if ($case.name -eq 'retraction' -and $report.rl_validation.status -ne 'passed') { throw 'Retraction RL checks missing' }
        if ($case.name -eq 'forceps-control' -and ($report.max_node_displacement_m -gt .0005 -or $report.forceps_grasp_lift_passed)) {
            throw 'Negative control moved tissue or falsely passed grasp'
        }
        $results+=@{case=$case.name;status='passed';report=$path;wall_seconds=$timer.Elapsed.TotalSeconds;source_scene_sha256=$report.source_scene_sha256}
        Write-Host "$($case.name): passed"
    }
    $hashes=@($results | ForEach-Object { $_.source_scene_sha256 } | Select-Object -Unique)
    if ($hashes.Count -ne 1) { throw 'Suite cases used different source scenes' }
    @{status='passed';appendectomy_completed=$false;clinical_validation=$false;timing_includes_startup=$true;cases=$results} |
        ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 'out/ct_appendectomy/suite/report.json'
} catch {
    @{status='failed';appendectomy_completed=$false;cases=$results;error=$_.Exception.Message} |
        ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 'out/ct_appendectomy/suite/report.json'
    throw
} finally { Pop-Location }
