#!/bin/zsh
# Build PyTorch 2.11.0 from source on Apple Silicon, with the MPS STFT + allocator
# patches needed for Applio / RVC training.
#
# Usage:
#   1. Clone PyTorch v2.11.0:
#        git clone --depth 1 --branch v2.11.0 https://github.com/pytorch/pytorch.git
#        cd pytorch && git submodule update --init --recursive --depth 1
#   2. Apply the patch from this repo:
#        git apply /path/to/applio-apple-silicon/pytorch-2.11.0.patch
#   3. Run this script from the PyTorch checkout:
#        /path/to/applio-apple-silicon/build-pytorch.sh
#
# Wheel lands in /tmp/pytorch-wheels/ (override with WHEEL_OUT=/dest).
# Build time: ~15 min cold, minutes with ccache warm.
#
# Target Python: 3.12 (matches Applio).

set -e

PYTORCH_SRC="${PYTORCH_SRC:-$PWD}"
WHEEL_OUT="${WHEEL_OUT:-/tmp/pytorch-wheels}"
PY="${PY:-python3.12}"

if [[ ! -f "$PYTORCH_SRC/setup.py" ]]; then
  echo "error: $PYTORCH_SRC doesn't look like a PyTorch checkout (no setup.py)" >&2
  echo "cd into the checkout first, or set PYTORCH_SRC=/path/to/pytorch" >&2
  exit 1
fi

for t in cmake ninja ccache; do
  command -v $t >/dev/null || { echo "error: $t not installed. Try: brew install cmake ninja ccache" >&2; exit 1; }
done

cd "$PYTORCH_SRC"

# Slim build: disable everything not needed for Apple Silicon training.
export USE_MPS=1
export USE_CUDA=0
export USE_DISTRIBUTED=0
export BUILD_TEST=0
export USE_KINETO=0
export USE_MKLDNN=0
export USE_QNNPACK=0
export USE_XNNPACK=0
export USE_NNPACK=0
export USE_FBGEMM=0
export USE_GLOO=0
export USE_TENSORPIPE=0
export USE_BREAKPAD=0
export USE_OPENMP=0

export MAX_JOBS="${MAX_JOBS:-$(sysctl -n hw.ncpu)}"

# Intentionally NOT setting CMAKE_OSX_ARCHITECTURES: doing so triggers
# PyTorch's cross-compile branch that forces host protoc to universal
# x86_64+arm64, which fails because recent Xcode CLT no longer ships
# x86_64 C++ stdlib stubs. Default arch on Apple Silicon is arm64.
unset CMAKE_OSX_ARCHITECTURES

export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export PYTORCH_BUILD_VERSION=2.11.0
export PYTORCH_BUILD_NUMBER=1

# CMake 4.x removed compat with cmake_minimum_required < 3.5, but PyTorch's
# bundled protobuf submodule still declares that. Allow the old policy.
export CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# Some Command Line Tools installs have libc++ headers only in the SDK subtree,
# not at the bin-adjacent path. Point clang at the SDK copy if default is bare.
SDK_LIBCXX=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
if [[ -d "$SDK_LIBCXX" ]] && [[ ! -f /Library/Developer/CommandLineTools/usr/include/c++/v1/string ]]; then
  export CPLUS_INCLUDE_PATH="$SDK_LIBCXX"
fi

# Shell scripts in some submodules lose exec permissions on clone.
find "$PYTORCH_SRC/scripts" -name "*.sh" -exec chmod +x {} +
find "$PYTORCH_SRC/third_party" -name "*.sh" -exec chmod +x {} + 2>/dev/null || true

mkdir -p "$WHEEL_OUT"
rm -rf "$PYTORCH_SRC/build" "$PYTORCH_SRC/build_host_protoc"

echo ">>> Building PyTorch 2.11.0 (MPS-only, slim) with $MAX_JOBS jobs..."
"$PY" -m pip wheel . --no-build-isolation --no-deps -w "$WHEEL_OUT"

echo
echo ">>> Done. Wheel:"
ls -lh "$WHEEL_OUT"/torch-2.11.0-*.whl
echo
echo ">>> Install into your Applio venv:"
echo "    /path/to/Applio/.venv/bin/pip install --force-reinstall --no-deps $WHEEL_OUT/torch-2.11.0-*.whl"
