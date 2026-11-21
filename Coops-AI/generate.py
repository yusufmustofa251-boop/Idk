import argparse, torch
from common import CKPT, CHARS, STOI, VOCAB
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("prompt", nargs="?", default="Indonesia adalah")
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--temp", type=float, default=0.8)
a = ap.parse_args()

model = GPT(VOCAB)
model.load_state_dict(torch.load(CKPT)["model"])
model.eval()
ids = torch.tensor([[STOI[c] for c in a.prompt if c in STOI] or [0]])
out = model.generate(ids, a.n, a.temp)[0].tolist()
print("".join(CHARS[i] for i in out))
