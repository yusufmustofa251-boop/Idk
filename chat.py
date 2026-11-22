import argparse, os, torch
from bpe import BPE
from model2 import GPT2

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="ckpt_chat.pt")
ap.add_argument("--temp", type=float, default=0.7)
ap.add_argument("--max", type=int, default=200, help="maks token jawaban")
a = ap.parse_args()

print("=== Coops-AI Chat ===", flush=True)
if not os.path.exists(a.ckpt):
    raise SystemExit(f"{a.ckpt} belum ada. Latih dulu (lihat panduan tahap chat).")
print("Loading model...", flush=True)
tok = BPE.load()
ck = torch.load(a.ckpt)
model = GPT2(**ck["cfg"])
model.load_state_dict(ck["model"])
model.eval()
stop = tok.specials["<|akhir|>"]
print(f"Siap (step {ck['step']}). Ketik 'keluar' buat berhenti. Jawabanku bisa salah ya!\n", flush=True)
while True:
    try:
        q = input("Kamu: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if q.lower() in ("keluar", "exit", "quit"):
        break
    if not q:
        continue
    ids = torch.tensor([tok.encode(f"<|pengguna|>{q}<|coops|>")])
    out = model.generate(ids, a.max, a.temp, 40, stop=stop)[0].tolist()[ids.shape[1]:]
    if out and out[-1] == stop:
        out = out[:-1]
    print("Coops:", tok.decode(out), "\n", flush=True)
