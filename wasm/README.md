# wasm/ — shared browser-build tooling

Both apps' single-file pages (`admm/wasm/build.sh`, `dw/wasm/build.sh`) use:

- `build/` and `build_wheel.sh`: the DP kernels in Cython (`build/dp_kernels.pyx`),
  cross-compiled to a Pyodide wasm32 wheel (Docker). A page embeds the wheel if
  one is in `build/dist/`, and runs pure Python otherwise.
- `keep_names.py`: the names the page builder must not rename when it
  obfuscates a bundle.

`admm/wasm/README.md` describes the pipeline and the measurements.
