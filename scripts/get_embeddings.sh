#!/usr/bin/env bash
# Fetch the public R3D embeddings of EchoNet-Dynamic (78 MB) into data/.
set -euo pipefail
mkdir -p data
curl -L -o data/echonet_embeddings.txt \
  https://raw.githubusercontent.com/Team-Echo-MIT/r3d-v0-embeddings/main/echonet_embeddings.txt
wc -l data/echonet_embeddings.txt
echo "Now place EchoNet-Dynamic FileList.csv in data/ (Stanford AIMI DUA) and run:"
echo "  echoaudit run --embeddings data/echonet_embeddings.txt --filelist data/FileList.csv --out results/"
