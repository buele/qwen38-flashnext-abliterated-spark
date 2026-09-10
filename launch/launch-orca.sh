#!/usr/bin/env bash
# Qwen3.8-Flash-Next ORCA(abliterated) on DGX Spark — contract-preserving launcher
# 契约:port 8000 / api-key YOUR_API_KEY_HERE / 模型名 aeon-uncensored
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="myllmbox/qwen38-flash-next-vllm:v2"
PORT="${PORT:-8000}"
MODEL_HOST="/home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
MODEL_DIR="/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
NAME="${NAME:-orca-flashnext}"
API_KEY="YOUR_API_KEY_HERE"

NST=$(find "$MODEL_HOST" -maxdepth 1 -name '*.safetensors' ! -name 'model-plefp8-*' | wc -l)
if [ "$NST" -lt 25 ]; then echo "✗ safetensors $NST/25(17主+8PLE+MTP)"; exit 1; fi
if [ -n "$(find "$MODEL_HOST" -name '*.part' -o -name '*.incomplete' | head -1)" ]; then echo "✗ 未完成文件"; exit 1; fi
[ -f "$MODEL_HOST/ple-nvfp4-00001-of-00008.safetensors" ] || { echo "✗ 缺 PLE 分片"; exit 1; }

docker rm -f "$NAME" >/dev/null 2>&1 || true
t=0; while :; do
  avail=$(LC_ALL=C free -g | awk '/^Mem:/{print $7}')
  [ "${avail:-0}" -ge 100 ] && break
  [ "$t" -ge 180 ] && { echo "✗ only ${avail}G"; exit 1; }
  [ "$t" = 0 ] && echo "  · waiting memory (${avail}G → 100G)…"
  sleep 5; t=$((t+5))
done

find "$MODEL_HOST" -maxdepth 1 -type f -name "*.safetensors" -exec dd if={} iflag=nocache count=0 status=none \; 2>/dev/null || true

echo "· launching $NAME on :$PORT (first boot ~12-15 min)"
docker run -d --name "$NAME" --gpus all --ipc=host \
  --cpuset-cpus "5-9,15-19" \
  -p "$PORT:8000" \
  -v /home/mwyzs/models:/models \
  -v /home/mwyzs/flashnext-cache:/cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer-workspace \
  -e VLLM_CACHE_ROOT=/cache/vllm-cache \
  -e MBX_PLE_MMAP=1 \
  -e MBX_PLE_MMAP_MODE=auto \
  -e MBX_PLE_MMAP_PREWARM=auto \
  -e MBX_PLE_MMAP_TAG=orca47 \
  -e MBX_PLE_MMAP_DIR=/cache/ple-mmap \
  -e OMP_NUM_THREADS=8 \
  -e MBX_VOCAB_GEMV=1 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e MALLOC_MMAP_THRESHOLD_=65536 \
  -e MALLOC_TRIM_THRESHOLD_=131072 \
  --entrypoint vllm "$IMAGE" serve "$MODEL_DIR" \
  --port 8000 \
  --served-model-name aeon-uncensored \
  --api-key "$API_KEY" \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.62 \
  --kv-cache-memory 5000000000 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --moe-backend humming \
   \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --async-scheduling \
  --trust-remote-code \
  --host 0.0.0.0

echo "· waiting health (25 min cap)…"
for i in $(seq 1 300); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "✓ READY :$PORT (aeon-uncensored, orca abliterated)"; exit 0; }
  docker ps -q --filter "name=$NAME" | grep -q . || { echo "✗ died — docker logs $NAME"; exit 1; }
  sleep 5
done
echo "✗ not healthy — docker logs $NAME"; exit 1
