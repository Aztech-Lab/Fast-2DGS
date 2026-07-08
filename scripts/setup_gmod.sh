#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -d gmod ]; then
  git clone https://github.com/Aztech-Lab/gmod.git
fi
cd gmod
pip install -e . --no-build-isolation
cd "$ROOT"
echo "gmod installed. Test with: python main_demo.py"