#!/usr/bin/env bash
#
# Rebuild the ADMM app's in-browser testbed from src/hemspolicy and admm/.
#
#   admm/wasm/build.sh            # bundle -> page -> single-file standalone
#   admm/wasm/build.sh --serve    # ...then serve it on :8765
#
# If wasm/build/dist/ holds a Pyodide wasm32 wheel (see wasm/build_wheel.sh) it is
# embedded too, and the page installs it with micropip before running the
# bundle. The Python then dispatches its DP recursions to the compiled kernels
# instead of numpy. Without a wheel everything still works, just slower.
#
# Three steps, each runnable on its own:
#   make_bundle.py   src/hemspolicy + admm    -> hemspolicy_bundle.py
#   make_page.py     admm/gui/index.html      -> hems_policy.html
#   make_standalone  page + bundle            -> hems_policy_standalone.html
#
# The last step lives in the sibling wasm_batt_optimizer project, which is
# where that tooling was written; this only supplies its two inputs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || PY=python3

STANDALONE="${MAKE_STANDALONE:-$ROOT/../wasm_batt_optimizer/make_standalone.py}"

"$PY" "$HERE/make_bundle.py"
"$PY" "$HERE/make_page.py"

if [ -x "$STANDALONE" ]; then
  # make_standalone renames every FunctionDef, which includes methods and
  # @property accessors -- and a method is reached through an Attribute node
  # that the renamer does not rewrite, so renaming one breaks it at runtime.
  # keep_names.py lists them; everything else (72 of 73 module-level
  # functions, i.e. the solver internals) still gets obfuscated.
  KEEP=()
  while read -r n; do KEEP+=(--keep "$n"); done < <("$PY" "$ROOT/wasm/keep_names.py" "$HERE/hemspolicy_bundle.py")
  echo "[keep] protecting ${#KEEP[@]} class members from renaming"

  WHL=$(ls "$ROOT"/wasm/build/dist/*pyodide*wasm32.whl 2>/dev/null | head -1)
  if [ -n "$WHL" ]; then
    echo "[wheel] embedding $(basename "$WHL")"
    WHEEL_ARG=(--wheel "$WHL")
  else
    echo "[wheel] none in wasm/build/dist; pure-Python page (run wasm/build_wheel.sh for kernels)"
    WHEEL_ARG=(--no-wheel)
  fi

  "$STANDALONE" "${WHEEL_ARG[@]}" --keep _api "${KEEP[@]}" \
      --html "$HERE/hems_policy.html" \
      --py   "$HERE/hemspolicy_bundle.py" \
      --out  "$HERE/hems_policy_standalone.html"
else
  echo "!! make_standalone.py not found at $STANDALONE" >&2
  echo "   set MAKE_STANDALONE=/path/to/make_standalone.py to inline the payload." >&2
  echo "   hems_policy.html still works if you serve this directory." >&2
fi

if [ "${1:-}" = "--serve" ]; then
  echo "http://127.0.0.1:8765/hems_policy_standalone.html"
  exec "$PY" -m http.server -d "$HERE" 8765
fi
