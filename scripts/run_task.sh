#!/usr/bin/env bash
# run_task.sh — Run a single pre-generated task config file locally.
#
# Usage:
#   ./scripts/run_task.sh <task_config.json> [output.db]
#
# Examples:
#   ./scripts/run_task.sh infra/configs/task_0.json
#   ./scripts/run_task.sh infra/configs/task_0.json my_result.db

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="$REPO_ROOT/target/release/oouccd-rs"
VENV_PYTHON="$REPO_ROOT/seq-venv/bin/python3"

# ── Preflight checks ─────────────────────────────────────────────────────────
if [ ! -f "$BINARY" ]; then
    echo "ERROR: binary not found at $BINARY" >&2
    echo "       Run: source seq-venv/bin/activate && cargo build --release" >&2
    exit 1
fi

if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: seq-venv not found." >&2
    exit 1
fi

# ── Python environment setup ──────────────────────────────────────────────────
VENV_SITE="$(find "$REPO_ROOT/seq-venv/lib" -maxdepth 2 -name "site-packages" | head -1)"
export PYTHONPATH="$VENV_SITE"

if [[ "$(uname)" == "Darwin" ]]; then
    FW_DIR="$("$VENV_PYTHON" -c "
import sys, os
p = sys.base_prefix
for _ in range(10):
    if os.path.basename(p) == 'Frameworks':
        print(p)
        break
    p = os.path.dirname(p)
" 2>/dev/null || true)"
    [ -n "$FW_DIR" ] && export DYLD_FRAMEWORK_PATH="$FW_DIR"
fi

# ── Args ─────────────────────────────────────────────────────────────────────
TASK_CONFIG="${1:?Usage: $0 <task_config.json> [output.db]}"

# Auto-generate DB name from config if not specified
if [ -n "${2:-}" ]; then
    DB="$2"
else
    LABEL="$("$VENV_PYTHON" -c "
import json, re
d = json.load(open('$TASK_CONFIG'))
label = d['method']['label']
label = re.sub(r'[ ()\-]+', '_', label).strip('_')
label = re.sub(r'_+', '_', label)
name = d.get('task_name', 'task')
print(f'{name}_{label}')
" 2>/dev/null || basename "$TASK_CONFIG" .json)"
    DB="$REPO_ROOT/${LABEL}.db"
fi

echo "[run_task] config : $TASK_CONFIG"
echo "[run_task] output : $DB"
echo ""

"$BINARY" --config-filepath "$TASK_CONFIG" --db-filepath "$DB"
