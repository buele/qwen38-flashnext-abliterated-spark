#!/bin/bash
# Start vLLM Qwen3.6-27B-AEON-Ultimate XS with DFlash n=12
# No reasoning_parser — raw output goes to content field directly

docker rm -f vllm-qwen35b 2>/dev/null
mkdir -p /models/vllm-cache

exec docker run -d --name vllm-qwen35b \
  --restart unless-stopped \
  --network host \
  --gpus all \
  --ipc host \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_MATMUL_PRECISION=high \
  -e NVIDIA_FORWARD_COMPAT=1 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e ENABLE_NVFP4_SM100=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass \
  -v /models/aeon-ultimate-xs:/models/xs:ro \
  -v /models/dflash-drafter:/models/dflash-drafter:ro \
  -v /models/vllm-cache:/root/.cache:rw \
  ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  -c "exec vllm serve /models/xs \
    --served-model-name aeon-ultimate qwen36-ultimate aeon-fast aeon-deep aeon-ultimate-xs qwen35b-heretic \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --dtype auto \
    --quantization modelopt \
    --kv-cache-dtype auto \
    --max-model-len 256000 \
    --max-num-seqs 64 \
    --max-num-batched-tokens 32768 \
    --gpu-memory-utilization 0.75 \
    --enable-chunked-prefill \
    --no-enable-prefix-caching \
    --generation-config vllm \
    --load-format safetensors \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --attention-backend flash_attn \
    --limit-mm-per-prompt '{\"image\": 4, \"video\": 2}' \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --mm-shm-cache-max-object-size-mb 256 \
    --speculative-config '{\"method\":\"dflash\",\"model\":\"/models/dflash-drafter\",\"num_speculative_tokens\":12}'"
