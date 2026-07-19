<#
Run the isolated basic test without using pytest's shared default temp folder.

Usage (after activating career-agent):
    .\scripts\run_basic_test.ps1
#>
param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$baseTemp = Join-Path $projectRoot (".pytest-tmp-" + [guid]::NewGuid().ToString("N"))

& $Python -m pytest tests\test_basic_e2e.py -q --basetemp $baseTemp
exit $LASTEXITCODE
