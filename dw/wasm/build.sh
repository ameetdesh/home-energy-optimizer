#!/usr/bin/env bash
#
# Build the single-file, in-browser version of the Dantzig-Wolfe app.
#
#   dw/wasm/build.sh            # bundle -> page -> multi_device_optimizer_standalone.html
#   dw/wasm/build.sh --serve    # ...then serve it on :8767
#
# Same pipeline as admm/wasm/build.sh for the ADMM app:
#   make_bundle.py        dw/ + src/home_energy_optimizer  -> dw_bundle.py (one flat module)
#   make_page.py          dw/gui/index.html     -> dw_page.html (Pyodide worker boot)
#   make_standalone.py    page + bundle (+ wheel) -> one HTML file, Python
#                         renamed, stripped, zlib+base64'd; kernels wheel inlined
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/bin/python"; [ -x "$PY" ] || PY=python3
STANDALONE="${MAKE_STANDALONE:-$ROOT/tools/make_standalone.py}"
OUT="$HERE/multi_device_optimizer_standalone.html"

"$PY" "$HERE/make_bundle.py"
"$PY" "$HERE/make_page.py"

[ -x "$STANDALONE" ] || { echo "!! make_standalone.py not found at $STANDALONE" >&2; exit 1; }

# Names reached through an attribute must survive renaming (see wasm/keep_names.py).
KEEP=()
while read -r n; do KEEP+=(--keep "$n"); done < <("$PY" "$ROOT/wasm/keep_names.py" "$HERE/dw_bundle.py")
echo "[keep] protecting ${#KEEP[@]} names from renaming"

WHL=$(ls "$ROOT"/wasm/build/dist/*pyodide*wasm32.whl 2>/dev/null | head -1 || true)
if [ -n "$WHL" ]; then
  echo "[wheel] embedding $(basename "$WHL")"; WHEEL_ARG=(--wheel "$WHL")
else
  echo "[wheel] none in wasm/build/dist; pure-Python page (run wasm/build_wheel.sh for kernels)"; WHEEL_ARG=(--no-wheel)
fi

"$STANDALONE" "${WHEEL_ARG[@]}" --keep _api "${KEEP[@]}" \
    --html "$HERE/dw_page.html" --py "$HERE/dw_bundle.py" --out "$OUT"

if [ "${1:-}" = "--serve" ]; then
  echo "serving on http://127.0.0.1:8767/$(basename "$OUT")"
  exec "$PY" -m http.server -d "$HERE" 8767
fi
