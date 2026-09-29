# Cross-compilation image for building Pyodide (Emscripten/WASM) wheels.
#
# The Emscripten version is pinned by pyodide-build and must line up with the
# Pyodide runtime loaded from the CDN in the standalone HTML:
#
#   Pyodide CDN | pyodide-build | Emscripten | Python
#   0.26.2      | 0.26.2        | 3.1.58     | 3.12
#   0.27.x      | 0.27.x        | 3.1.68+    | 3.12
#
# wheel is pinned <0.45: newer releases dropped the wheel.cli API that
# auditwheel-emscripten depends on.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake build-essential xz-utils ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir "wheel<0.45" && \
    pip install --no-cache-dir pyodide-build==0.26.2 cython==3.2.4 numpy==2.5.2 && \
    git clone --depth 1 https://github.com/emscripten-core/emsdk.git /opt/emsdk && \
    cd /opt/emsdk && ./emsdk install 3.1.58 && ./emsdk activate 3.1.58 && \
    pyodide xbuildenv install 0.26.2

ENV PATH="/opt/emsdk:/opt/emsdk/upstream/emscripten:${PATH}"
ENV EMSDK="/opt/emsdk"
WORKDIR /src
