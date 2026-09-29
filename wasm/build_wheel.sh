#!/usr/bin/env bash
#
# Cross-compile wasm/build/dp_kernels.pyx to a Pyodide wasm32 wheel.
#
#   wasm/build_wheel.sh            # build the wheel into wasm/build/dist/
#   wasm/build_wheel.sh --image    # force a rebuild of the builder image
#
# Reuses a pyodide-builder image if one is already present (the same image
# tools/make_standalone.py --build-wheel builds); the Dockerfile is in build/.
#
# The Emscripten/pyodide-build versions must line up with the Pyodide runtime
# the page loads from the CDN (0.26.2) -- see the Dockerfile.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"
IMAGE="pyodide-builder:0.26.2"

command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }

if [ "${1:-}" = "--image" ] || ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[wheel] building image $IMAGE (slow the first time)"
  docker build -t "$IMAGE" "$BUILD"
fi

# BUILD is bind-mounted, so setuptools intermediates survive between runs and
# will happily relink a stale .o - silently ignoring an edited .pyx.
rm -rf "$BUILD/dist" "$BUILD/build" "$BUILD/_skbuild"

echo "[wheel] cross-compiling with pyodide build"
docker run --rm -v "$BUILD:/src" "$IMAGE" bash -lc \
  "source /opt/emsdk/emsdk_env.sh >/dev/null 2>&1; cd /src && pyodide build --exports pyinit"

WHL=$(ls "$BUILD"/dist/*pyodide*wasm32.whl 2>/dev/null | head -1)
[ -n "$WHL" ] || { echo "no wasm32 wheel produced" >&2; exit 1; }
echo "[wheel] $WHL  ($(wc -c < "$WHL") bytes)"
