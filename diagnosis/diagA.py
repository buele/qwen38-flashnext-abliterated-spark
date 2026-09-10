import json, urllib.request, urllib.error, hashlib, os, subprocess

KEY="YOUR_API_KEY_HERE"; BASE="http://127.0.0.1:8000"
HDR={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"}

def post(path, body, timeout=120):
    req=urllib.request.Request(BASE+path, data=json.dumps(body).encode(), headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")

print("===== [1] /v1/completions raw =====")
st, d = post("/v1/completions", {"model":"aeon-uncensored","prompt":"The capital of France is","max_tokens":10,"temperature":0,"logprobs":5})
print(st, json.dumps(d)[:1200])

print("===== [2] chat with logprobs =====")
st, d = post("/v1/chat/completions", {"model":"aeon-uncensored","messages":[{"role":"user","content":"Hi"}],"max_tokens":8,"temperature":0,"logprobs":True,"top_logprobs":5})
print(st)
ch=d.get("choices",[{}])[0] if isinstance(d,dict) else {}
print("message:", json.dumps(ch.get("message",{}))[:500])
print("logprobs sample:", json.dumps((ch.get("logprobs") or {}).get("content",[])[:6])[:1500])

print("===== [3] tokenizer file hashes =====")
BASEM="/home/mwyzs/models"
for tag,d_ in [("ORCA","Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4"),("H47","Qwen3.8-Flash-Next-hibrid47"),("FP8","Qwen3.8-Flash-Next-Uncensored-NVFP4-FP8PLE")]:
    for fn in ["tokenizer.json","vocab.json","merges.txt","tokenizer_config.json"]:
        p=os.path.join(BASEM,d_,fn)
        if os.path.exists(p):
            h=hashlib.md5(open(p,'rb').read()).hexdigest()[:12]
            print(f"  {tag:5} {fn:22} {h}  {os.path.getsize(p)}")
        else:
            print(f"  {tag:5} {fn:22} MISSING")

print("===== [4] ple-mmap cache stat =====")
pm="/home/mwyzs/flashnext-cache/ple-mmap"
for f in sorted(os.listdir(pm)):
    p=os.path.join(pm,f); st_=os.stat(p)
    import time as _t
    print(f"  {f}  {st_.st_size/2**30:.2f}G  mtime={_t.strftime('%m-%d %H:%M',_t.localtime(st_.st_mtime))}")

print("===== [5] model file mtimes (orca) =====")
od=os.path.join(BASEM,"Qwen3.8-Flash-Next-Uncensored-NVFP4-PLE-NVFP4")
import time as _t
for f in ["ple-nvfp4-00001-of-00008.safetensors","ple-nvfp4-00002-of-00008.safetensors","model-mtp.safetensors","config.json","model.safetensors.index.json","model-00012-of-00017.safetensors","model-00013-of-00017.safetensors"]:
    p=os.path.join(od,f)
    if os.path.exists(p):
        st_=os.stat(p); print(f"  {f:42} mtime={_t.strftime('%m-%d %H:%M',_t.localtime(st_.st_mtime))}")
    else: print(f"  {f:42} MISSING")

print("===== [6] docker logs: ple/mmap lines =====")
os.system("docker logs orca-test 2>&1 | grep -iE 'ple|mmap|ngram' | grep -v 'Unrecognized' | head -15")
