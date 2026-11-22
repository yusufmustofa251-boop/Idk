"""Ambil teks siap pakai dari dataset Hugging Face (tanpa scraping, tanpa kena 429 Wikipedia).
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
                            "offset": offset, "length": PAGE}, tries=10)


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
