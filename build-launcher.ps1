$ErrorActionPreference='Stop'
$compiler=Join-Path $env:WINDIR 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
if (-not (Test-Path -LiteralPath $compiler)) { throw 'Windows .NET Framework C# compiler not found.' }
Push-Location $PSScriptRoot
try {
    & $compiler /nologo /target:winexe /reference:System.Windows.Forms.dll /out:IsaacSimulator.exe IsaacLauncher.cs
    if ($LASTEXITCODE) { throw 'Isaac launcher compilation failed.' }
    & $compiler /nologo /target:winexe /reference:System.Windows.Forms.dll /out:RobotAgent.exe IsaacLauncher.cs
    if ($LASTEXITCODE) { throw 'Robot agent launcher compilation failed.' }
} finally { Pop-Location }
