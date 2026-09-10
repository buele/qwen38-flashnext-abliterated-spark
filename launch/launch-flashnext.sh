#!/usr/bin/env bash
# Qwen3.8-Flash-Next (hibrid47) on DGX Spark — contract-preserving launcher
# 契约（不可变）：port 8000 / api-key YOUR_API_KEY_HERE / 模型名 aeon-uncensored(+别名)
# 基于 bilikaz recipe.yaml v2 配置，适配我们的老契约。
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="myllmbox/qwen38-flash-next-vllm:v2"
PORT=8000
MODEL_DIR="/home/mwyzs/models/Qwen3.8-Flash-Next-hibrid47"
NAME="qwen38-flashnext"
API_KEY="YOUR_API_KEY_HERE"

# --- 下载完整性校验：40 个文件齐 + index 存在 ---
if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "✗ checkpoint 不完整（缺 index）— 先跑 dl_hibrid47.sh"; exit 1
fi
NST=$(find "$MODEL_DIR" -maxdepth 1 -name '*.safetensors' | wc -l)
if [ "$NST" -lt 21 ]; then
  echo "✗ safetensors 只有 $NST/21"; exit 1
fi
if [ -n "$(find "$MODEL_DIR" -name '*.incomplete' | head -1)" ]; then
  echo "✗ 存在 .incomplete 文件 — 下载未完成"; exit 1
fi

# --- 内核页压缩提示（root 可选优化，不阻塞）---
cp_now=$(cat /proc/sys/vm/compaction_proactiveness 2>/dev/null || echo "?")
[ "$cp_now" != 0 ] && echo "  ⚠ vm.compaction_proactiveness=$cp_now (0 可 +10% 吞吐, 需 sudo, 跳过)"

# --- 内存门控：老容器先停，等统一内存回来 ---
OLD=$(docker ps -q --filter "name=qwen38-uncensored")
if [ -n "$OLD" ]; then
  echo "· stopping old container qwen38-uncensored"
  docker stop qwen38-uncensored >/dev/null
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true

t=0; while :; do
  avail=$(LC_ALL=C free -g | awk '/^Mem:/{print $7}')
  [ "${avail:-0}" -ge 100 ] && { echo "  ✓ memory: ${avail}G available"; break; }
  [ "$t" -ge 180 ] && { echo "  ✗ only ${avail}G after 180s (need 100G)"; exit 1; }
  [ "$t" = 0 ] && echo "  · waiting for memory (${avail}G → need 100G)…"
  sleep 5; t=$((t+5))
done

# --- 页缓存清理：把 checkpoint 文件逐出 page cache（无 root）---
find "$MODEL_DIR" -type f -name "*.safetensors" -exec dd if={} iflag=nocache count=0 status=none \; 2>/dev/null || true

echo "· launching $NAME on :$PORT (first boot ~12 min)"
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
  --entrypoint vllm "$IMAGE" serve "/models/Qwen3.8-Flash-Next-hibrid47" \
  --port 8000 \
  --served-model-name aeon-uncensored \
  --api-key "$API_KEY" \
  --distributed-executor-backend mp \
  --gpu-memory-utilization 0.70 \
  --kv-cache-memory 7000000000 \
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

echo "· waiting for health (up to 25 min)…"
for i in $(seq 1 300); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "✓ READY on :$PORT  (model: aeon-uncensored, key unchanged)"
    exit 0
  fi
  if ! docker ps -q --filter "name=$NAME" | grep -q .; then
    echo "✗ container died — docker logs $NAME"; exit 1
  fi
  sleep 5
done
echo "✗ not healthy after 25 min — docker logs $NAME"; exit 1
