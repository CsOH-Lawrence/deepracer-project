$ErrorActionPreference = 'Stop'
$drSyncPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$drSyncScript = Join-Path $PSScriptRoot 'sync_logs.py'
& $drSyncPython -u $drSyncScript --config (Join-Path $PSScriptRoot 'config.json')
