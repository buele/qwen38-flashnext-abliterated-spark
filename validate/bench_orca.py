#!/usr/bin/env python3
"""Benchmark TTFT / prefill / decode / concurrency for the Spark 8000 endpoint.

This vLLM build emits thinking tokens in delta.reasoning (not reasoning_content),
so we track first-thinking-token and first-content-token separately. Requests that
burn all tokens on thinking (finish=length, no content) are reported, not crashed.

Usage: ORCA_KEY=<key> python3 bench_orca.py
Env: ORCA_KEY, ORCA_BASE (default http://127.0.0.1:8000), ORCA_MODEL (default aeon-uncensored)
"""
import os, json, time, threading, urllib.request, statistics

KEY   = os.environ.get("ORCA_KEY",  "YOUR_API_KEY_HERE")
BASE  = os.environ.get("ORCA_BASE", "http://127.0.0.1:8000")
MODEL = os.environ.get("ORCA_MODEL", "aeon-uncensored")

PROMPT_SHORT  = "从1数到10，用逗号分隔。"
PROMPT_DECODE = "写一个连续的科幻故事，从人类第一次接触外星文明开始，尽量写长。"
LONG_TEXT     = "量子计算利用量子力学原理进行计算。量子比特可以处于叠加态，" * 120
PROMPT_PREFILL = LONG_TEXT + "\n\n用一句话总结上面的内容。"
N_CONCURRENT  = 4

def post_stream(prompt, max_tokens, temperature=0.7, seed=42):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": temperature,
        "seed": seed,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body, headers={
        "Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_think = t_content = None
    usage = None; finish = None; n_chunks = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "): continue
            p = line[6:]
            if p == "[DONE]": break
            try: d = json.loads(p)
            except Exception: continue
            if d.get("usage"): usage = d["usage"]
            ch = (d.get("choices") or [{}])[0]
            if ch.get("finish_reason"): finish = ch["finish_reason"]
            delta = ch.get("delta") or {}
            if delta.get("reasoning") and t_think is None:
                t_think = time.perf_counter() - t0
            if delta.get("content") and t_content is None:
                t_content = time.perf_counter() - t0
            n_chunks += 1
    total = time.perf_counter() - t0
    ctoks = (usage or {}).get("completion_tokens") or n_chunks
    ptoks = (usage or {}).get("prompt_tokens") or 0
    return {"t_think": t_think, "t_content": t_content, "total": total,
            "ctoks": ctoks, "ptoks": ptoks, "finish": finish}

def med(vals): return statistics.median(vals)

# ---- 1) TTFT: first thinking token & first content token, short prompt, 5 runs ----
tt = []
for i in range(5):
    r = post_stream(PROMPT_SHORT, 128)
    tt.append(r)
    print(f"ttft run{i+1}: think={r['t_think'] and round(r['t_think']*1000)}ms "
          f"content={r['t_content'] and round(r['t_content']*1000)}ms finish={r['finish']} ctok={r['ctoks']}", flush=True)

# ---- 2) decode throughput: 384 tok budget (room for thinking), 3 runs ----
dec = []
for i in range(3):
    r = post_stream(PROMPT_DECODE, 384)
    first = r["t_think"] or r["t_content"] or 0
    gen_time = r["total"] - first
    tps = r["ctoks"] / gen_time if gen_time > 0 else 0
    dec.append({"tok": r["ctoks"], "gen_s": gen_time, "tok_s": tps, "finish": r["finish"]})
    print(f"decode run{i+1}: {r['ctoks']} tok in {gen_time:.1f}s gen = {tps:.1f} tok/s (finish={r['finish']})", flush=True)

# ---- 3) prefill throughput: ~5k tok prompt, 3 runs (salted to defeat prefix cache) ----
pre = []
for i in range(3):
    p = LONG_TEXT + f"\n\n（第{i}组，校验码{i*7919}）请用一句话总结上面的内容。"
    r = post_stream(p, 256)
    first = r["t_think"] or r["t_content"] or 0
    pre_tps = r["ptoks"] / first if first else 0
    pre.append({"ptoks": r["ptoks"], "ttft_ms": first*1000, "prefill_tok_s": pre_tps})
    print(f"prefill run{i+1}: {r['ptoks']} prompt tok, first tok {first*1000:.0f} ms = {pre_tps:.0f} tok/s", flush=True)

# ---- 4) concurrent decode: 4 parallel streams, 3 runs ----
conc = []
for i in range(3):
    results: list = [None]*N_CONCURRENT
    def worker(idx):
        results[idx] = post_stream(PROMPT_DECODE, 384)
    t0 = time.perf_counter()
    ths = [threading.Thread(target=worker, args=(k,)) for k in range(N_CONCURRENT)]
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.perf_counter() - t0
    good = [r for r in results if r]
    toks = sum(r["ctoks"] for r in good)
    conc.append({"wall_s": wall, "tokens": toks, "agg_tok_s": toks/wall})
    print(f"concurrent run{i+1}: {N_CONCURRENT} streams x {toks//len(good)} tok, wall {wall:.1f}s = {toks/wall:.1f} tok/s aggregate", flush=True)

out = {
    "endpoint": BASE, "model": MODEL, "n_concurrent": N_CONCURRENT,
    "ttft_short": {
        "first_thinking_token_ms": {"runs": [round(r["t_think"]*1000) if r["t_think"] else None for r in tt],
                                     "median": round(med([r["t_think"]*1000 for r in tt if r["t_think"]])) if any(r["t_think"] for r in tt) else None},
        "first_content_token_ms": {"runs": [round(r["t_content"]*1000) if r["t_content"] else None for r in tt],
                                    "median": round(med([r["t_content"]*1000 for r in tt if r["t_content"]])) if any(r["t_content"] for r in tt) else None},
    },
    "decode": {"runs": dec, "median_tok_s": round(med([r["tok_s"] for r in dec]), 1)},
    "prefill": {"runs": pre, "median_tok_s": round(med([r["prefill_tok_s"] for r in pre]))},
    "concurrent": {"runs": conc, "median_agg_tok_s": round(med([r["agg_tok_s"] for r in conc]), 1)},
}
print("\n==== BENCH RESULT ====")
print(json.dumps(out, ensure_ascii=False, indent=1))
open(os.environ.get("BENCH_OUT", "/tmp/bench-orca.json"), "w").write(json.dumps(out, ensure_ascii=False, indent=1))
