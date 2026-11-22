"""Coops-AI updater: jalanin sekali buat nimpa semua file kode ke versi terbaru.

    python update.py

- Yang diupdate: kode (.py, run_loop.sh, requirements.txt) di folder tempat update.py ini berada
- Yang GAK disentuh: ckpt.pt (otak), folder data/ (corpus), sources.txt (kalau udah ada)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = {
    'common.py': r'''import os, re

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
''',
    'model.py': r'''import torch, torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, block=256, layers=4, heads=4, dim=256):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(block, dim)
        self.blocks = nn.Sequential(*[Block(dim, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=40):
        for _ in range(n):
            logits, _ = self(idx[:, -self.block:])
            logits = logits[:, -1] / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx
''',
    'train.py': r'''import argparse, os, time, numpy as np, torch
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
''',
    'generate.py': r'''import argparse, os, torch
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
''',
    'crawler.py': r'''import argparse, hashlib, os, time, requests
from urllib import robotparser
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from common import CORPUS, SEEN, clean

# Ganti pakai email/GitHub lu:
#   export CRAWLER_UA="Coops-AI/0.1 (emaillu@gmail.com)"
UA = {"User-Agent": os.environ.get(
    "CRAWLER_UA", "Coops-AI/0.1 (personal learning project; python-requests)")}
MIN_LEN = 400
DELAY = 3.0
SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".pdf", ".zip",
            ".mp4", ".mp3", ".css", ".js", ".ico", ".xml", ".rss", ".apk")


def get(url, params=None, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
        except Exception as e:
            print(f"   ! koneksi gagal ({e}), coba lagi {i+1}/{tries}", flush=True)
            if i < tries - 1:
                time.sleep(3 * (i + 1))
            continue
        if r.status_code == 200:
            return r
        print(f"   ! HTTP {r.status_code} dari {urlparse(url).netloc}", flush=True)
        if i < tries - 1:
            time.sleep(5 * (i + 1))
    return None


def load_seen():
    if not os.path.exists(SEEN):
        return set()
    return set(open(SEEN).read().split())


def save_doc(text, seen):
    text = clean(text)
    if len(text) < MIN_LEN:
        return 0
    h = hashlib.md5(text.encode()).hexdigest()
    if h in seen:
        return 0
    seen.add(h)
    with open(CORPUS, "a") as f:
        f.write(text + "\n\n")
    with open(SEEN, "a") as f:
        f.write(h + "\n")
    return len(text)


def wikipedia(lang, count, seen):
    # API Wikipedia cuma ngasih teks LENGKAP buat 1 artikel per request
    print(f"[1/2] Wikipedia ({lang}): {count} artikel acak", flush=True)
    total = added = 0
    for i in range(1, count + 1):
        r = get(f"https://{lang}.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "generator": "random",
            "grnnamespace": 0, "grnlimit": 1, "prop": "extracts",
            "explaintext": 1, "exlimit": 1})
        if r is None:
            time.sleep(DELAY * 2)
            continue
        try:
            pages = r.json().get("query", {}).get("pages", {})
        except ValueError:
            print("   ! respon bukan JSON", flush=True)
            continue
        for page in pages.values():
            n = save_doc(page.get("extract", ""), seen)
            if n:
                added += 1
                total += n
        if i % 10 == 0 or i == count:
            print(f"  {i}/{count} dicek | {added} masuk | +{total} karakter", flush=True)
        time.sleep(DELAY)
    return total


def allowed(url, robots):
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    if base not in robots:
        rp = robotparser.RobotFileParser()
        try:
            r = requests.get(base + "/robots.txt", headers=UA, timeout=15)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            elif r.status_code in (401, 403):
                rp = None
            else:
                rp.parse([])
        except Exception:
            rp = None
        if rp is not None:
            rp.modified()
        robots[base] = rp
    rp = robots[base]
    return rp is not None and rp.can_fetch(UA["User-Agent"], url)


def crawl_site(seed, limit, seen, robots):
    """Mulai dari 1 URL, ikutin link di domain yang sama sampai 'limit' halaman."""
    host = urlparse(seed).netloc
    queue, visited, fetched, total = [seed], {seed}, 0, 0
    while queue and fetched < limit:
        url = queue.pop(0)
        if not allowed(url, robots):
            print("  skip (robots.txt):", url[:70], flush=True)
            continue
        r = get(url, tries=1)
        time.sleep(DELAY)
        if r is None or "text/html" not in r.headers.get("Content-Type", ""):
            continue
        fetched += 1
        soup = BeautifulSoup(r.text, "html.parser")
        text = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        n = save_doc(text, seen)
        total += n
        print(f"  [{fetched}/{limit}] {url[:55]} | " + (f"+{n}" if n else "lewat"), flush=True)
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"]).split("#")[0]
            p = urlparse(link)
            if (p.scheme in ("http", "https") and p.netloc == host and link not in visited
                    and not p.path.lower().endswith(SKIP_EXT)):
                visited.add(link)
                queue.append(link)
    return total


def web(sources_file, pages, seen):
    seeds = []
    if os.path.exists(sources_file):
        for line in open(sources_file):
            line = line.strip()
            if line.startswith("http"):
                seeds.append(line)
    print(f"[2/2] Website lain ({sources_file}): {len(seeds)} situs, maks {pages} halaman per situs", flush=True)
    total, robots = 0, {}
    for seed in seeds:
        print(f"  situs: {seed}", flush=True)
        total += crawl_site(seed, pages, seen, robots)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", type=int, default=0, help="jumlah artikel Wikipedia (0 = skip)")
    ap.add_argument("--lang", default="id")
    ap.add_argument("--sources", default="sources.txt")
    ap.add_argument("--pages", type=int, default=30, help="maks halaman per situs")
    a = ap.parse_args()
    print("=== Coops-AI Crawler ===", flush=True)
    seen = load_seen()
    print(f"Dokumen yang udah pernah diambil: {len(seen)}", flush=True)
    n = wikipedia(a.lang, a.wiki, seen) if a.wiki else 0
    n += web(a.sources, a.pages, seen)
    size = os.path.getsize(CORPUS) if os.path.exists(CORPUS) else 0
    print(f"Selesai! +{n} karakter baru | corpus sekarang {size/1e6:.2f} MB", flush=True)
''',
    'import_hf.py': r'''"""Ambil teks siap pakai dari dataset Hugging Face (tanpa scraping, tanpa kena 429 Wikipedia).
Default: Wikipedia Indonesia (teks bersih). Cuma butuh 'requests', gak perlu install datasets.

Contoh:
    python import_hf.py --mb 10
"""
import argparse, random, time
from common import CORPUS
from crawler import save_doc, load_seen, get

API = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # maksimal baris per request

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="wikimedia/wikipedia")
ap.add_argument("--config", default="20231101.id")
ap.add_argument("--split", default="train")
ap.add_argument("--field", default="text", help="nama kolom yang isinya teks")
ap.add_argument("--mb", type=float, default=5, help="target tambahan corpus (MB)")
ap.add_argument("--delay", type=float, default=1.0)
a = ap.parse_args()

print("=== Coops-AI Import Dataset ===", flush=True)
print(f"Sumber: {a.dataset} [{a.config}] | target +{a.mb} MB", flush=True)
seen = load_seen()


def fetch(offset):
    return get(API, params={"dataset": a.dataset, "config": a.config, "split": a.split,
                            "offset": offset, "length": PAGE}, tries=3)


r = fetch(0)
if r is None:
    raise SystemExit("Gagal konek ke datasets-server. Cek nama --dataset/--config, atau dataset-nya butuh login (gated).")
total_rows = r.json().get("num_rows_total", 100000)
print(f"Total baris di dataset: {total_rows}", flush=True)

added_chars, n_docs, req = 0, 0, 0
used = set()
resp = r
while added_chars < a.mb * 1e6 and req < 3000:
    if resp is None:
        resp = fetch(random.randrange(0, max(total_rows - PAGE, 1), PAGE))
        if resp is None:
            time.sleep(5)
            continue
    req += 1
    try:
        rows = resp.json().get("rows", [])
    except ValueError:
        rows = []
    got = 0
    for item in rows:
        n = save_doc(item.get("row", {}).get(a.field, "") or "", seen)
        if n:
            got += 1
            added_chars += n
    n_docs += got
    print(f"  request {req} | +{got} dokumen | total +{added_chars/1e6:.2f} MB", flush=True)
    resp = None
    # lompat ke blok acak yang belum dipakai biar topiknya beragam
    time.sleep(a.delay)
print(f"Selesai! +{n_docs} dokumen, +{added_chars/1e6:.2f} MB masuk ke {CORPUS}", flush=True)
''',
    'run_loop.sh': r'''#!/bin/bash
# crawl -> train -> ulang terus. Jalanin: nohup ./run_loop.sh > log.txt 2>&1 &
round=1
while true; do
  echo ""
  echo "######## RONDE $round ########"
  python -u crawler.py --wiki 0 --pages 60
  python -u train.py --steps 500
  round=$((round+1))
done
''',
    'requirements.txt': r'''torch
numpy
requests
beautifulsoup4
''',
}

OPTIONAL = {"sources.txt": "# Tempel URL situs bahasa Indonesia di sini, SATU URL PER BARIS (tanpa tanda #).\n# Crawler mulai dari URL itu, terus ikutin link di situs yang sama.\n# robots.txt dicek otomatis: situs yang ngelarang bakal di-skip.\n"}
IGNORE = ["__pycache__/", "ckpt.pt", "ckpt.pt.tmp", "crawl.txt", "train.txt", "log.txt", "import.txt", "nohup.out"]

print("=== Coops-AI Updater ===")
print("Folder target:", HERE)
changed = 0
for name, content in FILES.items():
    path = os.path.join(HERE, name)
    old = open(path).read() if os.path.exists(path) else None
    if old == content:
        print(f"  = {name} (udah terbaru)")
        continue
    with open(path, "w") as f:
        f.write(content)
    if name.endswith(".sh"):
        os.chmod(path, 0o755)
    changed += 1
    print(f"  + {name} ({'baru' if old is None else 'diupdate'})")

for name, content in OPTIONAL.items():
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        open(path, "w").write(content)
        print(f"  + {name} (baru)")

gi = os.path.join(HERE, ".gitignore")
cur = open(gi).read().split() if os.path.exists(gi) else []
add = [x for x in IGNORE if x not in cur]
if add:
    with open(gi, "a") as f:
        f.write("\n" + "\n".join(add) + "\n")
    print("  + .gitignore:", ", ".join(add))

print(f"Selesai! {changed} file kode diupdate. ckpt.pt dan data/ aman.")