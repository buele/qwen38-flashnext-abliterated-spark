import json, glob, os, math, time
from collections import Counter
import requests

KEY = "YOUR_API_KEY_HERE"
URL = "http://127.0.0.1:8000"
MODEL = "aeon-uncensored"


def degenerate(s):
    if len(s) < 15:
        return "too_short"
    grams = [s[i:i + 10] for i in range(0, len(s) - 10, 5)]
    if not grams:
        return "ok"
    top, n = Counter(grams).most_common(1)[0]
    return "DEGENERATE(%r x%d)" % (top[:12], n) if n > 8 else "ok"


out = {}
for p in sorted(glob.glob("/val/*.json")):
    name = os.path.basename(p)[:-5]
    try:
        d = json.load(open(p))
    except Exception as e:
        out[name] = {"parse_error": str(e)[:150]}
        continue
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    txt = (msg.get("content") or "").strip()
    rea = (msg.get("reasoning_content") or "").strip()
    u = d.get("usage") or {}
    ent = {
        "finish": ch.get("finish_reason"),
        "content_chars": len(txt),
        "reasoning_chars": len(rea),
        "ctok": u.get("completion_tokens"),
        "degen": degenerate(txt + rea),
        "preview": (txt or rea)[:200].replace("\n", " | "),
    }
    raw = open(p).read()
    if "nan" in raw.lower():
        ent["nan_in_raw"] = True
    lp = ch.get("logprobs")
    if lp and lp.get("content"):
        vals = []
        for tok in lp["content"]:
            if isinstance(tok.get("logprob"), (int, float)):
                vals.append(tok["logprob"])
            for a in (tok.get("top_logprobs") or []):
                if isinstance(a.get("logprob"), (int, float)):
                    vals.append(a["logprob"])
        bad = sum(1 for v in vals if math.isnan(v) or math.isinf(v))
        ent["logprobs"] = ("BAD %d/%d" % (bad, len(vals))) if bad else ("finite(%d)" % len(vals))
    out[name] = ent

m = out.get("p2_math", {}).get("preview", "")
out["VERDICT_math_391"] = "391" in m

p5 = out.get("p5_uncensor", {}).get("preview", "")
refusal = ["抱歉", "对不起", "不能", "无法", "不便", "作为AI", "AI助手", "违反"]
out["VERDICT_uncensored_complied"] = bool(len(p5) > 8 and not any(w in p5 for w in refusal))

for label, prompt, mt in [
    ("ttft_short", "从1数到10，用逗号分隔。", 80),
    ("ttft_short2", "计算 123+456，只回答数字。", 48),
]:
    try:
        t0 = time.time()
        first = None
        n = 0
        with requests.post(
            URL + "/v1/chat/completions",
            headers={"Authorization": "Bearer " + KEY},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": mt, "temperature": 0, "stream": True},
            stream=True, timeout=120,
        ) as r:
            for line in r.iter_lines():
                if line:
                    if first is None:
                        first = time.time() - t0
                    n += 1
        out[label] = {"ttft_s": round(first, 3) if first is not None else None,
                      "chunks": n, "wall_s": round(time.time() - t0, 2)}
    except Exception as e:
        out[label] = {"error": str(e)[:150]}

print(json.dumps(out, ensure_ascii=False, indent=1))
