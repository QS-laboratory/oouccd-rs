# Source this file before building/running the embedded-Python application:
#   source scripts/python_env.sh [path/to/python]
_oouccd_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_oouccd_python="${1:-${PYO3_PYTHON:-$_oouccd_root/.venv/bin/python}}"
if ! test -x "$_oouccd_python"; then
    echo "Python interpreter not found: $_oouccd_python" >&2
    return 1
fi
export PYO3_PYTHON="$("$_oouccd_python" -c 'import sys; print(sys.executable)')"
export PYTHONPATH="$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export OMP_DYNAMIC=FALSE
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
_oouccd_python_lib="$("$PYO3_PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
# Some Linux clusters supply the runtime library but omit its linker-name
# symlink. Create that name locally; never modify the system installation.
"$PYO3_PYTHON" - "$_oouccd_root" <<'PY'
import sys, sysconfig
from pathlib import Path
name = sysconfig.get_config_var('LDLIBRARY')
runtime = sysconfig.get_config_var('INSTSONAME') or name
directory = sysconfig.get_config_var('LIBDIR')
if name and runtime and directory:
    target = Path(directory) / runtime
    if target.exists():
        local = Path(sys.argv[1]) / '.paper-env/lib'
        local.mkdir(parents=True, exist_ok=True)
        link = local / name
        if not link.exists():
            try:
                link.symlink_to(target.resolve())
            except FileExistsError:
                pass
PY
export LIBRARY_PATH="$_oouccd_root/.paper-env/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
if test -n "$_oouccd_python_lib"; then
    export LD_LIBRARY_PATH="$_oouccd_python_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _oouccd_root _oouccd_python _oouccd_python_lib
