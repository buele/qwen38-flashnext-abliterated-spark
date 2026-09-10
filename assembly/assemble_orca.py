#!/usr/bin/env python3
"""组装 hibrid47-orca: lychee(orca abliterated)主权重 + NVFP4 PLE 表(我们量化)"""
import json, os, shutil, sys

LY = "/home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE"
PLE_SRC = "/home/mwyzs/models/ple-nvfp4-orca"
DST = "/home/mwyzs/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
KEY_P = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"

os.makedirs(DST, exist_ok=True)

# 1) 复制主权重 + 配置(硬链接省空间)
for f in os.listdir(LY):
    if f.startswith("model-plefp8-"):
        continue  # FP8 PLE 大分片不进新目录
    src, dst = os.path.join(LY, f), os.path.join(DST, f)
    if not os.path.exists(dst):
        os.link(src, dst) if os.stat(src).st_dev == os.stat(os.path.dirname(dst)).st_dev else shutil.copy2(src, dst)

# 2) 复制 NVFP4 PLE 分片(改名对齐 hibrid47 命名)
for i in range(1, 9):
    src = os.path.join(PLE_SRC, f"ple-nvfp4-{i:05d}-of-00008.safetensors")
    dst = os.path.join(DST, f"ple-nvfp4-{i:05d}-of-00008.safetensors")
    if not os.path.exists(dst):
        os.link(src, dst)

# 3) 改写 index:以 lychee index 为基底,删 FP8 PLE 键,加 nvfp4 键
idx = json.load(open(os.path.join(LY, "model.safetensors.index.json")))
wm = idx["weight_map"]
# 删 FP8 PLE 表键
fp8_keys = [k for k in wm if "ngram_embedding" in k and ("plefp8" in wm[k] or "shard_" in k)]
for k in fp8_keys:
    if "weight_packed" in k or "weight_scale" in k or k.endswith(".weight"):
        if "ngram_embedding" in k:
            del wm[k]
# 加 nvfp4 键(8 shards × packed/scales + global)
for i in range(8):
    fn = f"ple-nvfp4-{i+1:05d}-of-00008.safetensors"
    wm[f"{KEY_P}.nvfp4_shard_{i}.packed"] = fn
    wm[f"{KEY_P}.nvfp4_shard_{i}.scales"] = fn
wm[f"{KEY_P}.nvfp4_global"] = "ple-nvfp4-00001-of-00008.safetensors"
# ngram 元数据键(ngram_heads_offsets/sizes)应还在 model-00003 里(lychee 保留了),确认
for k in list(wm):
    if "ngram_heads" in k:
        print("metadata key kept:", k, "->", wm[k])
idx["weight_map"] = wm
# total_size 重算(粗略:主权重不变 - fp8表 + nvfp4表)
json.dump(idx, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=1)

# 4) config: 加 ple_quantization 声明(hibrid47 同款字段)
cfg = json.load(open(os.path.join(DST, "config.json")))
tc = cfg.get("text_config", cfg)
# 保留 orca 的 quantization_config(compressed-tensors 混合),叠加 ple 声明
tc["ple_quantization"] = {
    "format": "nvfp4", "block": 16,
    "rows": 320001536, "dim": 160,
    "shards": 8, "shard_rows": 40000192,
}
if "text_config" in cfg:
    cfg["text_config"] = tc
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=1)

print("ASSEMBLED:", DST)
print("nvfp4 keys:", sum(1 for k in wm if "nvfp4" in k))
print("plefp8 keys left:", sum(1 for k in wm if "plefp8" in k))
