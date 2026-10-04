param([string]$Python = (Join-Path $PSScriptRoot '../../work/venv/Scripts/python.exe'))
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python environment missing. Pass -Python with a prepared interpreter.' }
$env:WARP_CACHE_PATH = Join-Path $PSScriptRoot '.cache/warp'
& $Python (Join-Path $PSScriptRoot 'local_validation/validate_tissue.py')
if ($LASTEXITCODE -ne 0) { throw 'Tissue baseline failed; inspect validation/tissue-baseline.json and console output.' }
