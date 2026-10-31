#!/usr/bin/env bash
# SAM3 is loaded through Transformers; no upstream clone/submodule installation is required.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r requirements.txt
python -c 'from src.sam3_adapter import load_sam3_api; load_sam3_api(); print("Transformers SAM3 API available")'
