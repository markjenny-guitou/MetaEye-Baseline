param(
    [Parameter(Position=0)]
    [ValidateSet('doctor','validate-data','train','evaluate')]
    [string]$Action='doctor',
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$ExtraArgs
)
$ErrorActionPreference='Stop'
$pythonExe=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Missing .venv. See README.md for installation.' }
$env:PYTHONUTF8='1'
& $pythonExe (Join-Path $PSScriptRoot 'main.py') $Action @ExtraArgs
exit $LASTEXITCODE
