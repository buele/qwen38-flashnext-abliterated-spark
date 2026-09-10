#!/usr/bin/env bash
# 下载完成后的自动流水线:校验 → 量化 → 组装 → 试启动 8001
set -uo pipefail
exec > /home/mwyzs/pipeline.log 2>&1
echo "=== PIPELINE START $(date +%F_%T) ==="

# 1) SHA256 校验(lychee 自带)
cd /home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE
if [ -f SHA256SUMS.txt ]; then
  echo "--- sha256 check (model files) ---"
  # 只校验模型文件(跳过 .part)
  ok=0; bad=0
  while read -r h f; do
    [ -f "$f" ] || continue
    if echo "$h  $f" | sha256sum -c - >/dev/null 2>&1; then ok=$((ok+1)); else bad=$((bad+1)); echo "BAD: $f"; fi
  done < SHA256SUMS.txt
  echo "sha256: ok=$ok bad=$bad"
  [ "$bad" -gt 0 ] && { echo "HALT: sha mismatch"; exit 1; }
fi

# 2) PLE FP8→NVFP4 量化(CPU 流式,约1-2h;输出到 ple-nvfp4-orca/)
echo "--- quantize PLE ---"
python3 /home/mwyzs/quant_ple.py
[ $? -ne 0 ] && { echo "HALT: quant failed"; exit 1; }

# 3) 组装
echo "--- assemble ---"
python3 /home/mwyzs/assemble_orca.py
[ $? -ne 0 ] && { echo "HALT: assemble failed"; exit 1; }

# 4) 试启动 8001(NAME=orca-test PORT=8001)
echo "--- test-launch on 8001 ---"
NAME=orca-test PORT=8001 bash /home/mwyzs/launch-orca.sh
echo "=== PIPELINE DONE $(date +%F_%T) ==="
