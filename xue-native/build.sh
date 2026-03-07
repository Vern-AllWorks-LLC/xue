#!/bin/bash
# Build C extensions for xue (Python 3.12 and 3.14t)
# Cross-platform: Linux (x86_64, aarch64), macOS (x86_64, arm64)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
XUE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

# Configurable Python paths (override via env vars)
PYTHON312="${PYTHON312:-$XUE_ROOT/xue-python-install/bin/python3}"
PYTHON314="${PYTHON314:-$XUE_ROOT/xue-python-install-314/bin/python3}"

XUE_LIB312="$XUE_ROOT/xue-python/Lib/xue"
XUE_LIB312_INST="$XUE_ROOT/xue-python-install/lib/python3.12/xue"
XUE_LIB314="$XUE_ROOT/xue-python-install-314/lib/python3.14t/xue"

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

# Build for Python 3.12
if [ -x "$PYTHON312" ] && [ -d "$XUE_LIB312" ]; then
    build_ext "$PYTHON312" "$XUE_LIB312" "Python 3.12"
    # Also copy to installed location
    if [ -d "$XUE_LIB312_INST" ]; then
        cp "$XUE_LIB312"/_*_accel.cpython-312*.so "$XUE_LIB312_INST/" 2>/dev/null || true
        echo "  Copied to install: $XUE_LIB312_INST"
    fi
else
    echo "Skipping Python 3.12 (not found at $PYTHON312)"
fi

# Build for Python 3.14t
if [ -x "$PYTHON314" ] && [ -d "$XUE_LIB314" ]; then
    build_ext "$PYTHON314" "$XUE_LIB314" "Python 3.14t"
else
    echo "Skipping Python 3.14t (not found at $PYTHON314)"
fi

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
        [ -d "$XUE_LIB312" ] && cp "_xue_cuda_kernels${CUDA_EXT}" "$XUE_LIB312/"
        [ -d "$XUE_LIB314" ] && cp "_xue_cuda_kernels${CUDA_EXT}" "$XUE_LIB314/"
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
for label_python in "Python 3.12:$PYTHON312" "Python 3.14t:$PYTHON314"; do
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
