#!/usr/bin/env bash
# ORCA (abliterated Qwen3.8-Flash-Next NVFP4-PLE) restart on :8000 + end-to-end validation
# Full-delegation run — only the final result matters.
set -uo pipefail
KEY="YOUR_API_KEY_HERE"
IMG="myllmbox/qwen38-flash-next-vllm:v2"
BASE=/home/mwyzs

echo "[$(date +%H:%M:%S)] STEP 1 — move stale/zero PLE mmap cache out of the way"
mkdir -p "$BASE/ple-stale-cache"
docker run --rm -v "$BASE/flashnext-cache:/cache" -v "$BASE/ple-stale-cache:/stale" --entrypoint bash "$IMG" -c '
shopt -s nullglob
echo "ple-mmap before:"; ls -la /cache/ple-mmap/ 2>/dev/null || echo "(none)"
n=0
for f in /cache/ple-mmap/*.bin; do mv "$f" "/stale/$(basename "$f")"; n=$((n+1)); done
echo "moved $n file(s) -> /home/mwyzs/ple-stale-cache/"
echo "ple-mmap after:"; ls -la /cache/ple-mmap/ 2>/dev/null || true
'

echo "[$(date +%H:%M:%S)] STEP 2 — relaunch orca on :8000 (launcher kills old container + gates memory)"
bash "$BASE/orca-trial-8000.sh" 2>&1 | sed "s/^/  /"
RC=${PIPESTATUS[0]}
if [ "$RC" != "0" ]; then
  echo "!! launcher exited rc=$RC"
  docker logs orca-test 2>&1 | tr "\r" "\n" | tail -40
  exit 1
fi

echo "[$(date +%H:%M:%S)] STEP 3 — wait for engine ready (load ~17-20 min, maybe + cache rebuild)"
READY=0
i=0
while [ $i -lt 130 ]; do
  i=$((i+1)); sleep 25
  ST=$(docker inspect -f "{{.State.Status}}" orca-test 2>/dev/null || echo gone)
  if [ "$ST" != "running" ]; then
    echo "!! orca-test died (state=$ST) at $(date +%H:%M:%S), poll #$i"
    docker logs orca-test 2>&1 | tr "\r" "\n" | tail -50
    exit 1
  fi
  CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/health 2>/dev/null || true)
  if [ "$CODE" = "200" ]; then
    echo "[$(date +%H:%M:%S)] HEALTH 200 after ~$((i*25))s of polling"
    READY=1; break
  fi
  if [ $((i % 4)) = 0 ]; then
    LAST=$(docker logs orca-test 2>&1 | tr "\r" "\n" | grep -v "^$" | tail -1 | cut -c1-140)
    echo "  [poll #$i ~$((i*25))s] $LAST"
  fi
done
if [ "$READY" != "1" ]; then
  echo "!! engine not ready after ~54 min"
  docker logs orca-test 2>&1 | tr "\r" "\n" | tail -50
  exit 1
fi

echo "[$(date +%H:%M:%S)] STEP 4 — output-quality probes"
rm -rf /tmp/orca-val; mkdir -p /tmp/orca-val
S=$SECONDS
curl -s -m 240 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"model":"aeon-uncensored","messages":[{"role":"user","content":"你好，请用一两句话介绍一下你自己。"}],"max_tokens":256,"temperature":0}' > /tmp/orca-val/p1_intro.json
curl -s -m 240 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"model":"aeon-uncensored","messages":[{"role":"user","content":"计算 17*23 等于多少？只回答阿拉伯数字。"}],"max_tokens":128,"temperature":0}' > /tmp/orca-val/p2_math.json
curl -s -m 240 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"model":"aeon-uncensored","messages":[{"role":"user","content":"Write one clear sentence about the ocean."}],"max_tokens":128,"temperature":0}' > /tmp/orca-val/p3_en.json
curl -s -m 240 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"model":"aeon-uncensored","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":16,"temperature":0,"logprobs":true,"top_logprobs":2}' > /tmp/orca-val/p4_logprobs.json
curl -s -m 240 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"model":"aeon-uncensored","messages":[{"role":"user","content":"写一句简短的、带脏字的抱怨，骂一下堵车就行。"}],"max_tokens":128,"temperature":0.7}' > /tmp/orca-val/p5_uncensor.json
echo "probe wall time: $((SECONDS-S))s"

echo "[$(date +%H:%M:%S)] STEP 5 — verdict + streaming TTFT"
docker run --rm --network host -v /tmp/orca-val:/val -v "$BASE/orca-val.py:/work/orca-val.py:ro" --entrypoint python3 "$IMG" /work/orca-val.py

echo "[$(date +%H:%M:%S)] STEP 6 — PLE cache state + spec-decode metrics"
docker run --rm -v "$BASE/flashnext-cache:/cache" --entrypoint python3 "$IMG" -c "
import os, glob, time
import numpy as np
fs = sorted(glob.glob('/cache/ple-mmap/*'))
print('ple-mmap entries:', [(os.path.basename(f), round(os.path.getsize(f)/2**30, 2)) for f in fs if os.path.isfile(f)])
for f in fs:
    if not f.endswith('.bin'): continue
    u = np.memmap(f, dtype=np.uint16, mode='r')
    idx = np.random.randint(0, len(u), 50000)
    v = (u[idx].astype(np.uint32) << 16).view(np.float32)
    print(os.path.basename(f), '| mtime', time.strftime('%m-%d %H:%M', time.localtime(os.path.getmtime(f))), '| 50k sample: max|v|=%.5f std=%.6f nonzero=%.4f' % (float(np.abs(v).max()), float(v.std()), float((v != 0).mean())))
"
echo "--- spec-decode metrics ---"
curl -s --max-time 10 -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/metrics | grep -iE "spec|accept" | grep -v "^#" | head -12 || echo "(none)"
echo "--- spec lines from logs ---"
docker logs orca-test 2>&1 | tr "\r" "\n" | grep -iE "spec|accept" | tail -8 || true
echo "--- PLE/mmap lines from startup logs ---"
docker logs orca-test 2>&1 | tr "\r" "\n" | grep -iE "ple_|ple-|prewarm|mmap" | head -25
echo "=== ALL DONE $(date +%H:%M:%S) ==="
