
#!/usr/bin/env bash
# WiFi(66.160)专用下载器:curl 逐文件断点续传
# 用法: dl_wifi_lychee.sh [启动/继续]
set -u
SRC_IP=192.168.66.160
REPO="lychee888/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE"
DEST="/home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE"
LOG="/home/mwyzs/dl_lychee.log"
BASE="https://hf-mirror.com/$REPO/resolve/main"
mkdir -p "$DEST"
cd "$DEST" || exit 1

# 文件清单(SHA256SUMS 里的模型文件 + 关键配置)
FILES=(
  model.safetensors.index.json config.json generation_config.json
  chat_template.jinja tokenizer.json tokenizer_config.json vocab.json merges.txt
  preprocessor_config.json video_preprocessor_config.json
  model-mtp.safetensors
)
for i in $(seq 1 17); do
  FILES+=("$(printf 'model-%05d-of-00017.safetensors' $i)")
done
for i in $(seq 0 9); do
  FILES+=("$(printf 'model-plefp8-%05d.safetensors' $i)")
done

total=${#FILES[@]}
n=0
for f in "${FILES[@]}"; do
  n=$((n+1))
  # 已完成(存在且非临时)则跳过
  if [ -f "$f" ] && [ ! -f "$f.part" ]; then continue; fi
  echo "[$(date +%H:%M:%S)] $n/$total $f" >> "$LOG"
  for attempt in 1 2 3 4 5; do
    curl -sS -C - --interface $SRC_IP -o "$f.part" --max-time 7200 \
      --retry 3 --retry-delay 5 -L "$BASE/$f" >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then mv -f "$f.part" "$f"; echo "  done: $f" >> "$LOG"; break; fi
    echo "  attempt $attempt failed rc=$rc, retry" >> "$LOG"; sleep 5
  done
done
echo "EXIT $(date +%F_%T) files=$(ls -1 | grep -c safetensors)" >> "$LOG"
