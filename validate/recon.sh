#!/usr/bin/env bash
# Recon before restart: cache state, engine PLE code, fix persistence
echo "=== 1. PLE-MMAP dir: real vs apparent size (sparse check) ==="
ls -la /home/mwyzs/flashnext-cache/ple-mmap/ 2>/dev/null
for f in /home/mwyzs/flashnext-cache/ple-mmap/*; do
  [ -f "$f" ] && echo "REAL $(du -sh "$f" | cut -f1) | APPARENT $(du -sh --apparent-size "$f" | cut -f1) | $(basename "$f")"
done
echo
echo "=== 2. current orca-test startup lines re PLE/mmap/prewarm ==="
docker logs orca-test 2>&1 | tr "\r" "\n" | grep -iE "ple_|ple-|prewarm|mmap" | head -30
echo
echo "=== 3. h47 launcher (launch-flashnext.sh) env block ==="
grep -nE "MBX|PLE|CACHE|MODEL_DIR|served-model|PORT=" /home/mwyzs/launch-flashnext.sh
echo
echo "=== 4. image code: PLE mmap build/reuse logic ==="
docker run --rm --entrypoint bash myllmbox/qwen38-flash-next-vllm:v2 -c '
F=$(grep -rln --include="*.py" -e "PLE_MMAP" -e "ple-mmap" /usr/local/lib/python3*/dist-packages /usr/local/lib/python3*/site-packages /usr/lib/python3*/dist-packages /opt /workspace 2>/dev/null | sort -u | head -6)
echo "files: $F"
for x in $F; do
  echo "--- $x ---"
  grep -nE "PLE_MMAP|ple-mmap|isfile|exists|from_file|memmap|zeros|truncate|fallocate|PREWARM|prewarm|reuse" "$x" | head -40
done'
echo
echo "=== 5. fix3 persistence spot-check (scale_2 must be ~4e-5 multiplicative) ==="
docker run --rm -v /home/mwyzs/models:/models -v /tmp/recon_check.py:/work/recon_check.py:ro --entrypoint python3 myllmbox/qwen38-flash-next-vllm:v2 /work/recon_check.py
echo
echo "=== 6. remote helper scripts present ==="
ls -la /home/mwyzs/orca-trial-8000.sh /home/mwyzs/orca-val.py /home/mwyzs/restart-validate.sh 2>/dev/null
