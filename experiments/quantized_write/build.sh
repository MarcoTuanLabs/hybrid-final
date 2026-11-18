#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
for name in arc limited; do
    clang++ -O3 -std=c++17 -dynamiclib -framework Accelerate "$name.cpp" -o "build/$name.dylib"
done
