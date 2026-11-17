#!/bin/bash
# Build C extensions for xue (Python 3.15 and free-threaded 3.15t)
# Cross-platform: Linux (x86_64, aarch64), macOS (x86_64, arm64)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XUE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

# Interpreters to build for (override via env vars). Each must have the xue
# package in its standard library (a xue-python install or an assembled dist);
# the extensions are written next to it.
PYTHON315="${PYTHON315:-$XUE_ROOT/xue-python-install-315/bin/python3}"
PYTHON315T="${PYTHON315T:-$XUE_ROOT/xue-python-install-315t/bin/python3}"

# Detect architecture for SIMD flags
ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|amd64)  SIMD_FLAGS="-mavx2" ;;
    aarch64|arm64) SIMD_FLAGS="" ;;  # NEON is always available on aarch64
    *)             SIMD_FLAGS="" ;;
esac

# Detect OS for linker flags and compiler
OS_NAME="$(uname -s)"
case "$OS_NAME" in
    Linux)   LDFLAGS_EXTRA="-ldl" ; CC="${CC:-gcc}" ;;
    Darwin)  LDFLAGS_EXTRA=""     ; CC="${CC:-clang}" ;;
    MINGW*|MSYS*|CYGWIN*)
             LDFLAGS_EXTRA=""     ; CC="${CC:-gcc}" ;;
    *)       LDFLAGS_EXTRA=""     ; CC="${CC:-gcc}" ;;
esac

# macOS on Apple Silicon: -march=native works with clang
COMMON_CFLAGS="-O3 -march=native $SIMD_FLAGS -fPIC -Wall -Wno-unused-function"
SOURCES="_units_accel.c _autodiff_accel.c _tensor_accel.c"

build_ext() {
    local PYTHON="$1"
    local DEST="$2"
    local LABEL="$3"

    local INCLUDES="-I$($PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))")"
    local SUFFIX="$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")"

    echo "=== Building for $LABEL ==="
    echo "    Compiler: $CC ($ARCH)"
    echo "    SIMD: ${SIMD_FLAGS:-none (scalar fallback)}"
    echo "    Suffix: $SUFFIX"
    echo "    Dest: $DEST"

    for src in $SOURCES; do
        local name="${src%.c}"
        local out="${DEST}/${name}${SUFFIX}"
        echo "  Compiling $src -> $out"
        $CC -shared $COMMON_CFLAGS \
            $INCLUDES \
            "$src" -o "$out" \
            -lm $LDFLAGS_EXTRA
        echo "  OK: $(ls -la "$out" | awk '{print $5, $9}')"
    done
}

xue_dir() {
    "$1" -c "import os, sysconfig; print(os.path.join(sysconfig.get_path('stdlib'), 'xue'))"
}

XUE_DIRS=""
for label_python in "Python 3.15:$PYTHON315" "Python 3.15t:$PYTHON315T"; do
    label="${label_python%%:*}"
    python="${label_python#*:}"
    if [ -x "$python" ] && [ -d "$(xue_dir "$python")" ]; then
        build_ext "$python" "$(xue_dir "$python")" "$label"
        XUE_DIRS="$XUE_DIRS $(xue_dir "$python")"
    else
        echo "Skipping $label (no xue-enabled interpreter at $python)"
    fi
done

# Build CUDA kernels (optional — Linux/macOS only, requires nvcc)
if command -v nvcc &>/dev/null; then
    echo "=== Building CUDA kernels ==="
    case "$OS_NAME" in
        Linux)   CUDA_EXT=".so" ;;
        Darwin)  CUDA_EXT=".dylib" ;;
        *)       CUDA_EXT=".so" ;;
    esac
    nvcc -shared -O3 -Xcompiler -fPIC \
        _xue_cuda_kernels.cu -o "_xue_cuda_kernels${CUDA_EXT}" \
        -lcudart 2>/dev/null && {
        echo "  CUDA kernels built: _xue_cuda_kernels${CUDA_EXT}"
        for d in $XUE_DIRS; do cp "_xue_cuda_kernels${CUDA_EXT}" "$d/"; done
    } || echo "  CUDA build failed (no GPU hardware? continuing without GPU)"
else
    echo "=== Skipping CUDA (nvcc not found) ==="
fi

echo ""
echo "=== Build complete ==="
echo "    Platform: $OS_NAME $ARCH"
if [ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ]; then
    echo "    SIMD: AVX2"
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "    SIMD: NEON"
else
    echo "    SIMD: scalar (no hardware SIMD)"
fi

# Verify
for label_python in "Python 3.15:$PYTHON315" "Python 3.15t:$PYTHON315T"; do
    label="${label_python%%:*}"
    python="${label_python#*:}"
    [ -x "$python" ] || continue
    echo ""
    echo "Verifying $label:"
    $python -c "
from xue._units_accel import Dimension, Unit, Quantity, DimensionError
print('  _units_accel: OK')
from xue._autodiff_accel import Variable, grad, value_and_grad, jacobian
print('  _autodiff_accel: OK')
from xue._tensor_accel import Tensor, float32, float64, ShapeError, zeros, ones, eye, arange
print('  _tensor_accel: OK (CUDA:', 'yes' if hasattr(Tensor, 'to_gpu') else 'no', ')')
" 2>&1 || echo "  Verification failed"
done
