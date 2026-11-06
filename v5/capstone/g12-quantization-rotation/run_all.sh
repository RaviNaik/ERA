#!/usr/bin/env bash
# All four steps. Each is resumable: rerun this after an interruption and it continues.
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)/scripts"
"$D/01_setup.sh"
"$D/02_convert_quantize.sh"
"$D/03_evaluate.sh"
"$D/04_report.sh"
