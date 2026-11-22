"""Ambil sebagian teks web bahasa Indonesia dari CC-100 (file statis, bukan API).
Download-nya bisa DILANJUT (resume) kalau putus, jadi jauh lebih stabil.

    python import_cc100.py --mb 40
"""
import argparse, codecs, lzma, os, re, time
import requests
from common import CORPUS
from crawler import save_doc, load_seen, UA

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="https://data.statmt.org/cc-100/id.txt.xz")
ap.add_argument("--mb", type=float, default=40, help="target tambahan corpus (MB)")
ap.add_argument("--part", default="cc100_id.xz.part", help="file download sementara (bisa dilanjut)")
ap.add_argument("--step", type=float, default=8, help="MB terkompresi per putaran download")
ap.add_argument("--max-download", type=float, default=300, help="batas total download (MB)")
ap.add_argument("--retry-wait", type=float, default=5, help="jeda dasar (detik) sebelum nyoba ulang kalau putus")
a = ap.parse_args()

# filter kualitas: buang URL/email/nomor panjang, spam judi, dan paragraf yang bukan bahasa Indonesia
BAD = re.compile(r"https?://|www\.|\S+@\S+\.\S+|\d{9,}", re.I)
SPAM = re.compile(r"\b(slot|togel|judi|casino|poker|bokep|porno|sbobet|gacor|maxwin|bandar)\b", re.I)
STOP = {"yang", "dan", "di", "untuk", "dengan", "dari", "ini", "itu", "pada", "dalam",
        "tidak", "akan", "juga", "adalah", "ke", "oleh", "atau", "sebagai"}


def good(p):
    if len(p) < 60 or BAD.search(p) or SPAM.search(p):
        return False
    words = re.findall(r"[a-z]+", p.lower())
    if len(words) < 8:
        return False
    return sum(w in STOP for w in words) >= max(2, len(words) // 20)


def grow(target):
    """Download (lanjutan) sampai file .part >= target byte. Auto-retry kalau putus.
    Return True kalau berhasil, "eof" kalau file sumber udah habis, False kalau gagal terus."""
    fails = 0
    while True:
        done = os.path.getsize(a.part) if os.path.exists(a.part) else 0
        if done >= target:
            return True
        if fails >= 15:
            return False
        try:
            h = dict(UA)
            h["Range"] = f"bytes={done}-"
            with requests.get(a.url, headers=h, stream=True, timeout=(15, 60)) as r:
                if r.status_code == 416:
                    return "eof"
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                mode = "ab" if (r.status_code == 206 and done) else "wb"
                got = done if mode == "ab" else 0
                with open(a.part, mode) as f:
                    for chunk in r.iter_content(256 * 1024):
                        f.write(chunk)
                        got += len(chunk)
                        if got >= target:
                            break
        except Exception as e:
            # kalau tadi sempat dapet data, itu bukan "gagal" (koneksi cuma putus di tengah)
            fails = 0 if os.path.getsize(a.part) > done else fails + 1
            print(f"  ! putus ({str(e)[:50]}), lanjut lagi... ({fails}/15)", flush=True)
            time.sleep(min(a.retry_wait * (fails + 1), 30))
            continue
        if os.path.getsize(a.part) <= done:
            fails += 1


seen = load_seen()
pseen = set()
dec = codecs.getincrementaldecoder("utf-8")(errors="ignore")
xz = lzma.LZMADecompressor()
st = {"tail": "", "paras": [], "added": 0, "docs": 0}


def flush():
    if st["paras"]:
        n = save_doc("\n".join(st["paras"]), seen)
        if n:
            st["added"] += n
            st["docs"] += 1
        st["paras"] = []


def feed(raw):
    lines = (st["tail"] + dec.decode(raw)).split("\n")
    st["tail"] = lines.pop()
    for ln in lines:
        ln = ln.strip()
        if not ln:
            flush()
        elif good(ln):
            h = hash(ln)
            if h not in pseen:
                pseen.add(h)
                st["paras"].append(ln)


def decompress(data):
    global xz
    out = b""
    while data:
        try:
            out += xz.decompress(data)
        except lzma.LZMAError as e:
            print("  ! data rusak:", e, flush=True)
            break
        if xz.eof:
            data = xz.unused_data
            xz = lzma.LZMADecompressor()
        else:
            break
    return out


print("=== Coops-AI Import CC-100 ===", flush=True)
goal, step, cap = a.mb * 1e6, int(a.step * 1e6), int(a.max_download * 1e6)
print(f"Target: +{a.mb} MB teks bersih | file sementara: {a.part}", flush=True)
fed = 0
while True:
    size = os.path.getsize(a.part) if os.path.exists(a.part) else 0
    if size > fed:
        with open(a.part, "rb") as f:
            f.seek(fed)
            while fed < size:
                data = f.read(min(8_000_000, size - fed))
                if not data:
                    break
                fed += len(data)
                feed(decompress(data))
        print(f"  diproses {fed/1e6:.1f} MB | +{st['docs']} dokumen | +{st['added']/1e6:.2f} MB masuk corpus", flush=True)
    if st["added"] >= goal:
        break
    if fed >= cap:
        print("Batas download tercapai.", flush=True)
        break
    print("  download lanjut...", flush=True)
    res = grow(fed + step)
    if res == "eof":
        print("File sumber udah habis dibaca.", flush=True)
        if os.path.getsize(a.part) <= fed:
            break
    elif not res:
        print("! Gagal download terus. Coba lagi nanti, bakal dilanjut dari yang udah ada.", flush=True)
        break
flush()
size = os.path.getsize(CORPUS) if os.path.exists(CORPUS) else 0
print(f"Selesai! +{st['docs']} dokumen, +{st['added']/1e6:.2f} MB | corpus sekarang {size/1e6:.2f} MB", flush=True)
