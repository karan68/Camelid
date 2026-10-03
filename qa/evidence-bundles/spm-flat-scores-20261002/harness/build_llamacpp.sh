#!/usr/bin/env bash
# Build llama.cpp's llama-tokenize at the commit camelid's SPM port was verified against (acd79d603).
set -eu
DIR=/mnt/disks/data/research/llama.cpp-acd79d6
if [ ! -d "$DIR/.git" ]; then
  git clone -q https://github.com/ggml-org/llama.cpp "$DIR"
fi
cd "$DIR"
git fetch -q origin
git checkout -q acd79d603
git log --oneline -1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=OFF > /tmp/f2r/llamacpp-cmake.log 2>&1
echo "CMAKE_EXIT=$?"
cmake --build build --target llama-tokenize -j 4 > /tmp/f2r/llamacpp-build.log 2>&1
echo "BUILD_EXIT=$?"
ls -la build/bin/llama-tokenize
echo LLAMACPP_DONE
