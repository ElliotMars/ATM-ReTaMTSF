#!/usr/bin/env bash
set -euo pipefail
task_action=$1
shift
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$task_root"
task_python=${PIVOT_PYTHON:-"$task_root/.venv-pivot/bin/python"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export TOKENIZERS_PARALLELISM=false
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec "$task_python" -m pivot_stage2_v2.suite "$task_action" "$@"
