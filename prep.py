"""Tahap 1: latih tokenizer BPE dari corpus, terus ubah corpus jadi token (data/tokens.npy)."""
import argparse, os, time
import numpy as np
from common import CORPUS
from bpe import BPE, TOK_PATH

ap = argparse.ArgumentParser()
ap.add_argument("--vocab", type=int, default=4096)
ap.add_argument("--retrain", action="store_true", help="latih ulang tokenizer (ckpt2.pt lama jadi gak kepake!)")
a = ap.parse_args()

print("=== Coops-AI Prep ===", flush=True)
if not os.path.exists(CORPUS):
    raise SystemExit("Corpus belum ada. Isi dulu pake crawler.py / import_hf.py")
print("[1/3] Baca corpus...", flush=True)
text = open(CORPUS, encoding="utf-8", errors="ignore").read()
print(f"      {len(text)/1e6:.1f} juta karakter", flush=True)

if os.path.exists(TOK_PATH) and not a.retrain:
    print("[2/3] Tokenizer udah ada, dipake lagi (hemat waktu)", flush=True)
    tok = BPE.load()
else:
    print("[2/3] Latih tokenizer BPE (bisa beberapa menit)...", flush=True)
    t0 = time.time()
    tok = BPE.train(text, a.vocab, log=lambda s: print(s, flush=True))
    tok.save()
    print(f"      selesai {time.time()-t0:.0f}s | vocab {tok.vocab_size}", flush=True)

print("[3/3] Ubah corpus jadi token...", flush=True)
t0 = time.time()
ids = tok.encode(text)
arr = np.array(ids, dtype=np.uint16)
np.save("data/tokens.npy", arr)
print(f"Selesai! {len(arr)/1e6:.2f} juta token | {len(text)/len(arr):.1f} karakter/token | {time.time()-t0:.0f}s", flush=True)
