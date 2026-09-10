import inspect, re

def P(*a):
    print(*a, flush=True)

P("########## SCRIPT A: vLLM ModelOpt NVFP4 code inspection ##########")
from vllm.model_executor.layers.quantization import modelopt as M

src = inspect.getsource(M)
P("FILE:", M.__file__, "len:", len(src))

for m in re.finditer(r"^class (\w+)", src, re.M):
    P("class:", m.group(1))

def dump_class(name, limit=14000):
    m = re.search(r"class %s\b.*?(?=^class |\Z)" % re.escape(name), src, re.S | re.M)
    if not m:
        P("NOT FOUND:", name)
        return
    t = m.group(0)
    P("----- %s (%d chars) -----" % (name, len(t)))
    P(t[:limit])

dump_class("ModelOptNvFp4FusedMoEMethod")

P("\n===== input_scale / dynamic occurrences in modelopt.py =====")
for i, line in enumerate(src.splitlines(), 1):
    if re.search(r"input_scale|dynamic|static", line):
        P("%5d: %s" % (i, line.rstrip()[:160]))

P("\n===== FusedMoE layer.py: input_scale in load_weights =====")
from vllm.model_executor.layers import fused_moe as FM
fsrc = inspect.getsource(FM)
P("FILE:", FM.__file__)
for i, line in enumerate(fsrc.splitlines(), 1):
    if re.search(r"input_scale", line):
        P("%5d: %s" % (i, line.rstrip()[:160]))

P("\n===== FusedMoE layer class: input_scale param creation =====")
m = re.search(r"class FusedMoE\b.*?(?=^class |\Z)", fsrc, re.S | re.M)
if m:
    t = m.group(0)
    for i, line in enumerate(t.splitlines(), 1):
        if re.search(r"input_scale|torch\.empty|torch\.zeros|Parameter\(", line):
            P("%5d: %s" % (i, line.rstrip()[:170]))
