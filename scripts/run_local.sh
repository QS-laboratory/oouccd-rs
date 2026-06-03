#!/usr/bin/env bash
# run_local.sh — Generate task configs from a master template and run all locally.
#
# Usage:
#   ./scripts/run_local.sh <master_config.json>
#
# Example:
#   ./scripts/run_local.sh infra/config_templates/config_h2_074.json
#
# Output DBs are written to the repo root, named: <template_basename>_<method>.db

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
    echo "       Run: python3 -m venv seq-venv && seq-venv/bin/pip install pyscf openfermion numpy scipy" >&2
    exit 1
fi

# ── Python environment setup ──────────────────────────────────────────────────
VENV_SITE="$(find "$REPO_ROOT/seq-venv/lib" -maxdepth 2 -name "site-packages" | head -1)"
export PYTHONPATH="$VENV_SITE"

# macOS: locate the Python framework directory for dyld
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
MASTER_CONFIG="${1:?Usage: $0 <master_config.json>}"
BASE="$(basename "$MASTER_CONFIG" .json)"

# ── Step 1: Generate task configs ─────────────────────────────────────────────
echo ""
echo "[run_local] Generating configs from: $MASTER_CONFIG"
GEN_OUT="$("$VENV_PYTHON" "$REPO_ROOT/infra/generate_configs.py" --config "$MASTER_CONFIG")"
echo "[run_local] $GEN_OUT"
N="$(echo "$GEN_OUT" | grep -oE '[0-9]+' | head -1)"

# ── Step 2: Run each task ─────────────────────────────────────────────────────
for i in $(seq 0 $((N - 1))); do
    TASK="$REPO_ROOT/infra/configs/task_${i}.json"

    LABEL="$("$VENV_PYTHON" -c "
import json, re
d = json.load(open('$TASK'))
label = d['method']['label']
label = re.sub(r'[ ()\-]+', '_', label).strip('_')
label = re.sub(r'_+', '_', label)
print(label)
")"

    DB="$REPO_ROOT/${BASE}_${LABEL}.db"

    echo ""
    echo "[run_local] [$((i + 1))/$N] $LABEL → $(basename "$DB")"
    "$BINARY" --config-filepath "$TASK" --db-filepath "$DB"
done

echo ""
echo "[run_local] All done ✓  ($N tasks)"
