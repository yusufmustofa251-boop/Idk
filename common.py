import os, re

DATA = "data"
CORPUS = os.path.join(DATA, "corpus.txt")
SEEN = os.path.join(DATA, "seen.txt")
CKPT = "ckpt.pt"

# vocab dikunci biar checkpoint lama tetap valid walau data baru masuk
CHARS = "\n" + "".join(chr(i) for i in range(32, 127))
STOI = {c: i for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)

os.makedirs(DATA, exist_ok=True)


def clean(text: str) -> str:
    text = re.sub(r"={2,}.*?={2,}", " ", text)          # judul section wiki
    text = re.sub(r"\[\d+\]", "", text)                  # rujukan [1]
    text = "".join(c for c in text if c in STOI)         # buang karakter aneh
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()
