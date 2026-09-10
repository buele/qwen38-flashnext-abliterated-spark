import inspect, re, json, glob
import torch
torch.set_grad_enabled(False)

def P(*a):
    print(*a, flush=True)

P("########## C1: ModelOptNvFp4FusedMoE FULL SOURCE ##########")
from vllm.model_executor.layers.quantization import modelopt as M
src = inspect.getsource(M)
m = re.search(r"class ModelOptNvFp4FusedMoE\b.*?(?=^class |\Z)", src, re.S | re.M)
t = m.group(0) if m else "NOT FOUND"
P("len:", len(t))
P(t)

P("\n########## C2: raw lines 1050-1170 & 1230-1360 (NvFp4 linear methods) ##########")
lines = src.splitlines()
for lo, hi in [(1050, 1170), (1230, 1360)]:
    P("----- lines %d-%d -----" % (lo, hi))
    for i in range(lo - 1, min(hi, len(lines))):
        P("%5d: %s" % (i + 1, lines[i][:170]))

P("\n########## C3: PerTensorScaleParameter ##########")
try:
    from vllm.model_executor.layers.quantization.utils import quant_utils as QU
    s2 = inspect.getsource(QU)
    mm = re.search(r"class PerTensorScaleParameter\b.*?(?=^class |\Z)", s2, re.S | re.M)
    P(mm.group(0) if mm else "not in quant_utils")
except Exception as e:
    P("err1:", e)
    import subprocess
    r = subprocess.run(["grep", "-rn", "class PerTensorScaleParameter", "/usr/local/lib/python3.12/dist-packages/vllm/"], capture_output=True, text=True)
    P(r.stdout)

P("\n########## C4: ModelOptMixedPrecisionConfig + template matching ##########")
mm = re.search(r"class ModelOptMixedPrecisionConfig\b.*?(?=^class |\Z)", src, re.S | re.M)
P(mm.group(0) if mm else "NOT FOUND")

P("\n########## C5: who consumes quantized_layers / how .N. is matched ##########")
r = subprocess_run = None
import subprocess
r = subprocess.run(["grep", "-rn", "-e", "quantized_layers", "-e", r"\.N\.", "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py"], capture_output=True, text=True)
P(r.stdout[:8000])

P("\n########## C6: h47 config non-expert entries (the ~149) ##########")
hc = json.load(open("/models/Qwen3.8-Flash-Next-hibrid47/config.json"))
hq = hc["quantization_config"]["quantized_layers"]
others = [k for k in hq if "experts" not in k]
P("count non-expert:", len(others))
for k in sorted(others)[:60]:
    P("  ", k, "->", hq[k])

P("\n########## C7: PLE dequant corr (48 rows) ##########")
from safetensors import safe_open
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])

def deq(D, n=48):
    p = sorted(glob.glob(D + "/*ple*.safetensors"))[0]
    with safe_open(p, framework="pt") as f:
        ks = list(f.keys())
        pk = [k for k in ks if k.endswith(".packed")][0]
        sk = [k for k in ks if k.endswith(".scales")][0]
        gk = [k for k in ks if k.endswith("nvfp4_global")]
        packed = f.get_slice(pk)[0:n].clone()
        scales = f.get_slice(sk)[0:n].clone()
        g = f.get_tensor(gk[0]).item() if gk else 1.0
    pb = packed.view(torch.uint8)
    lo = FP4[pb & 0xF].float()
    hi = FP4[pb >> 4].float()
    vals = torch.stack([lo, hi], -1).reshape(n, -1)
    sc = scales.float().repeat_interleave(16, 1)
    return vals * sc * g

do = deq("/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4")
dh = deq("/models/Qwen3.8-Flash-Next-hibrid47")
a, b = do.flatten(), dh.flatten()
va, vb = a - a.mean(), b - b.mean()
P("PLE corr(orca,h47)=%.6f  orca amax=%.4f std=%.5f | h47 amax=%.4f std=%.5f" % (
    (va @ vb / (va.norm() * vb.norm())).item(), a.abs().max(), a.std(), b.abs().max(), b.std()))
P("per-row max abs diff: %.6g" % (a - b).abs().max().item())

P("\n########## C8: embed/lm_head/router/gate corr ##########")
ORCA = "/models/Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"
H47 = "/models/Qwen3.8-Flash-Next-hibrid47"
o_idx = json.load(open(ORCA + "/model.safetensors.index.json"))["weight_map"]
h_idx = json.load(open(H47 + "/model.safetensors.index.json"))["weight_map"]

def get(D, idx, k):
    with safe_open(D + "/" + idx[k], framework="pt") as f:
        return f.get_tensor(k)

PAIRS = ["model.language_model.embed_tokens.weight", "lm_head.weight",
         "model.language_model.layers.10.mlp.gate.weight",
         "model.language_model.layers.10.mlp.shared_expert.gate_proj.weight",
         "model.language_model.layers.0.mlp.gate.weight",
         "model.language_model.layers.47.mlp.gate.weight"]
for ok in PAIRS:
    ok2 = next((c for c in (ok, ok.replace("model.language_model.", "model.")) if c in o_idx), None)
    hk2 = next((c for c in (ok, ok.replace("model.language_model.", "model.")) if c in h_idx), None)
    if not ok2 or not hk2:
        P("  %-58s SKIP" % ok[:58])
        continue
    a = get(ORCA, o_idx, ok2).float().flatten()
    b = get(H47, h_idx, hk2).float().flatten()
    if a.shape != b.shape:
        P("  %-58s shape %s vs %s" % (ok[:58], a.shape, b.shape))
        continue
    va, vb = a - a.mean(), b - b.mean()
    corr = (va @ vb / (va.norm() * vb.norm())).item()
    P("  %-58s corr=%.5f md=%.3e" % (ok[:58], corr, (a - b).abs().mean().item()))
    del a, b
P("SCRIPT C DONE")
