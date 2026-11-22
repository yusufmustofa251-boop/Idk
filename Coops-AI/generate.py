import argparse, os, torch
from common import CKPT, CHARS, STOI, VOCAB
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?", default="Indonesia adalah")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--temp", type=float, default=0.8)
a = ap.parse_args()

print("=== Coops-AI Generate ===", flush=True)
if not os.path.exists(CKPT):
    raise SystemExit("Belum ada checkpoint. Train dulu: python train.py --steps 1000")
print("[1/2] Loading model...", flush=True)
ck = torch.load(CKPT)
model = GPT(VOCAB)
model.load_state_dict(ck["model"])
model.eval()
print(f"      model step {ck['step']}", flush=True)
print(f"[2/2] Nulis {a.n} karakter dari prompt: {a.prompt!r}\n", flush=True)
ids = torch.tensor([[STOI[c] for c in a.prompt if c in STOI] or [0]])
out = model.generate(ids, a.n, a.temp)[0].tolist()
print("".join(CHARS[i] for i in out))
