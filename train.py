import argparse, os, time, numpy as np, torch
from common import CORPUS, CKPT, STOI, VOCAB
from model import GPT

BLOCK, BATCH, LR = 256, 32, 6e-4

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1000)
a = ap.parse_args()

print("=== Coops-AI Trainer ===", flush=True)
threads = os.cpu_count() or 4
torch.set_num_threads(threads)
print(f"[1/4] Baca corpus... (pakai {threads} core)", flush=True)
if not os.path.exists(CORPUS):
    raise SystemExit("Corpus belum ada. Jalanin dulu: python crawler.py --wiki 20")
lut = np.zeros(256, dtype=np.uint8)
for c, i in STOI.items():
    lut[ord(c)] = i
raw = np.frombuffer(open(CORPUS, "rb").read(), dtype=np.uint8)
if len(raw) <= BLOCK * 4:
    raise SystemExit("Corpus masih terlalu kecil, jalanin crawler dulu.")
data = torch.from_numpy(lut[raw].astype(np.int64))
print(f"      corpus: {len(data)/1e6:.2f} juta karakter", flush=True)

print("[2/4] Bikin model...", flush=True)
model = GPT(VOCAB, block=BLOCK)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
print(f"      {sum(p.numel() for p in model.parameters())/1e6:.1f} juta parameter", flush=True)

print("[3/4] Cek checkpoint...", flush=True)
step = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    step = ck["step"]
    print(f"      lanjut dari step {step}", flush=True)
else:
    print("      belum ada, mulai dari nol", flush=True)


def save():
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, CKPT + ".tmp")
    os.replace(CKPT + ".tmp", CKPT)


def batch():
    # sampling acak dari SEMUA corpus -> data lama ikut keulang (anti lupa)
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    return x, y


print(f"[4/4] Mulai training {a.steps} step (Ctrl+C buat berhenti)...", flush=True)
t0 = time.time()
for n in range(1, a.steps + 1):
    x, y = batch()
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    step += 1
    if n == 1 or n % 10 == 0:
        el = time.time() - t0
        print(f"  step {step} ({n}/{a.steps}) | loss {loss.item():.3f} | {el/n:.2f}s/step", flush=True)
    if step % 200 == 0:
        save()
        print("  >> checkpoint kesimpen", flush=True)
save()
print(f"Selesai! Checkpoint kesimpen di step {step}", flush=True)
