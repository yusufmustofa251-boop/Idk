import argparse, os, time, numpy as np, torch
from common import CORPUS, CKPT, CHARS, STOI, VOCAB
from model import GPT

BLOCK, BATCH, LR = 256, 32, 6e-4
torch.set_num_threads(os.cpu_count() or 4)

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1000)
a = ap.parse_args()

lut = np.zeros(256, dtype=np.uint8)
for c, i in STOI.items():
    lut[ord(c)] = i
data = lut[np.frombuffer(open(CORPUS, "rb").read(), dtype=np.uint8)]
assert len(data) > BLOCK * 4, "corpus masih terlalu kecil, jalanin crawler dulu"
data = torch.from_numpy(data.astype(np.int64))

model = GPT(VOCAB, block=BLOCK)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
step = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    step = ck["step"]
    print(f"lanjut dari step {step}")


def save():
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, CKPT + ".tmp")
    os.replace(CKPT + ".tmp", CKPT)


def batch():
    # sampling acak dari SEMUA corpus -> data lama ikut keulang (anti lupa)
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    return x, y


t0 = time.time()
for _ in range(a.steps):
    x, y = batch()
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    step += 1
    if step % 50 == 0:
        print(f"step {step} | loss {loss.item():.3f} | {time.time()-t0:.0f}s", flush=True)
    if step % 200 == 0:
        save()
save()
print("checkpoint kesimpen, step", step)
