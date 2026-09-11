#!/usr/bin/env bash
# ORCA final launch — PORT 8000.
# Baseline: original proven config (cpuset 10 cores, OMP 8, image-default compilation).
# Change: KV 7G -> 15G + GMU 0.70 -> 0.78 (KV pool 842k tok = 3.2x 262k concurrency,
# eliminates KV-exhaustion queueing at 4+ streams measured 9/11).
set -euo pipefail

IMAGE="myllmbox/qwen38-flash-next-vllm:v2"
PORT=8000
MODEL_DIR="/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
NAME="orca-test"
API_KEY="YOUR_API_KEY_HERE"

if [ ! -f "/home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4/model.safetensors.index.json" ]; then
  echo "✗ orca index missing"; exit 1
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

t=0; while :; do
  avail=$(LC_ALL=C free -g | awk '/^Mem:/{print $7}')
  [ "${avail:-0}" -ge 100 ] && { echo "  ✓ memory: ${avail}G available"; break; }
  [ "$t" -ge 180 ] && { echo "  ✗ only ${avail}G after 180s (need 100G)"; exit 1; }
  [ "$t" = 0 ] && echo "  · waiting for memory (${avail}G → need 100G)…"
  sleep 5; t=$((t+5))
done

echo "· launching $NAME (final) on :$PORT"
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
  --gpu-memory-utilization 0.78 \
  --kv-cache-memory 15000000000 \
  --kv-cache-dtype fp8 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --async-scheduling \
  --trust-remote-code \
  --host 0.0.0.0

echo "· launched on :$PORT; watch: docker logs -f $NAME"
