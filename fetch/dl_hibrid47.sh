#!/usr/bin/env bash
# Spark 上后台下载 myllmbox/Qwen3.8-Flash-Next-hibrid47 (~99GB) via hf-mirror
# 只下载，不启动任何服务。可重复运行（断点续传）。
set -uo pipefail
REPO="myllmbox/Qwen3.8-Flash-Next-hibrid47"
DEST="/home/mwyzs/models/Qwen3.8-Flash-Next-hibrid47"
LOG="/home/mwyzs/dl_hibrid47.log"
mkdir -p "$DEST"
export HF_ENDPOINT=https://hf-mirror.com
export PATH="$HOME/.local/bin:$PATH"
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --local-dir "$DEST" >> "$LOG" 2>&1
  rc=$?
else
  # 没有 hf CLI 就用容器
  docker run --rm -e HF_ENDPOINT=https://hf-mirror.com \
    -v /home/mwyzs/models:/dl --entrypoint python3 \
    myllmbox/qwen38-flash-next-vllm:v2 \
    -c "from huggingface_hub import snapshot_download; snapshot_download('$REPO', local_dir='/dl/Qwen3.8-Flash-Next-hibrid47')" >> "$LOG" 2>&1
  rc=$?
fi
echo "EXIT_CODE=$rc $(date +%F_%T)" >> "$LOG"
