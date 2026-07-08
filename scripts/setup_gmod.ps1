# Clone and install the CUDA renderer (gmod). Run from repo root.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path "gmod")) {
    git clone https://github.com/Aztech-Lab/gmod.git
}
Set-Location gmod
pip install -e . --no-build-isolation
Set-Location $Root
Write-Host "gmod installed. Test with: python main_demo.py"