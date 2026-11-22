import argparse, math, os, time
import numpy as np, torch
from bpe import BPE
from model2 import GPT2

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--data", default="data/tokens.npy")
ap.add_argument("--out", default="ckpt2.pt")
ap.add_argument("--init", default=None, help="mulai dari bobot checkpoint ini (buat tahap chat)")
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--dim", type=int, default=384)
ap.add_argument("--heads", type=int, default=6)
ap.add_argument("--block", type=int, default=256)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--warmup", type=int, default=100)
a = ap.parse_args()

print("=== Coops-AI Trainer v2 ===", flush=True)
threads = os.cpu_count() or 4
torch.set_num_threads(threads)
tok = BPE.load()

print(f"[1/4] Baca data {a.data}... (pakai {threads} core)", flush=True)
if not os.path.exists(a.data):
    raise SystemExit("Data token belum ada. Jalanin: python prep.py (atau chat_data.py buat tahap chat)")
data = torch.from_numpy(np.load(a.data).astype(np.int64))
print(f"      {len(data)/1e6:.2f} juta token", flush=True)

cfg = dict(vocab=tok.vocab_size, block=a.block, layers=a.layers, heads=a.heads, dim=a.dim)
step, state, init = 0, None, None
print("[2/4] Siapin model...", flush=True)
if os.path.exists(a.out):
    state = torch.load(a.out)
    cfg, step = state["cfg"], state["step"]
    print(f"      lanjut dari {a.out} (step {step})", flush=True)
elif a.init:
    init = torch.load(a.init)
    cfg = init["cfg"]
    print(f"      mulai dari bobot {a.init} (step {init['step']})", flush=True)
else:
    print("      model baru", flush=True)
if cfg["vocab"] != tok.vocab_size:
    raise SystemExit("Ukuran tokenizer beda sama checkpoint. Jangan latih ulang tokenizer kalau udah punya ckpt.")
BLOCK = cfg["block"]
if len(data) < BLOCK * 8:
    raise SystemExit("Data masih terlalu sedikit.")
n_val = max(len(data) // 100, BLOCK + 2)
val, train = data[-n_val:], data[:-n_val]

model = GPT2(**cfg)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
if state:
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["opt"])
elif init:
    model.load_state_dict(init["model"])
print(f"      {sum(p.numel() for p in model.parameters())/1e6:.1f} juta parameter", flush=True)


def lr_at(s):
    return a.lr * min(1.0, (s + 1) / a.warmup)


def batch(d):
    ix = torch.randint(len(d) - BLOCK - 1, (a.batch,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + BLOCK + 1] for i in ix])
    return x, y


@torch.no_grad()
def val_loss():
    model.eval()
    ls = [model(*batch(val))[1].item() for _ in range(10)]
    model.train()
    return sum(ls) / len(ls)


def save():
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": cfg}, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)


print(f"[3/4] Mulai training {a.steps} step (Ctrl+C buat berhenti)...", flush=True)
t0 = time.time()
model.train()
for n in range(1, a.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step)
    x, y = batch(train)
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
        print(f"  >> val loss {val_loss():.3f} (checkpoint kesimpen)", flush=True)
        save()
save()
print(f"[4/4] Selesai! {a.out} kesimpen di step {step} | val loss {val_loss():.3f}", flush=True)
