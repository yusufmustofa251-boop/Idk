import argparse, hashlib, os, time, requests
from urllib import robotparser
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from common import CORPUS, SEEN, clean

# Wikipedia minta User-Agent yang jelas. Ganti pakai email/GitHub lu:
#   export CRAWLER_UA="Coops-AI/0.1 (emaillu@gmail.com)"
UA = {"User-Agent": os.environ.get(
    "CRAWLER_UA", "Coops-AI/0.1 (personal learning project; python-requests)")}
MIN_LEN = 400
DELAY = 1.0


def get(url, params=None, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30)
        except Exception as e:
            print(f"   ! koneksi gagal ({e}), coba lagi {i+1}/{tries}", flush=True)
            time.sleep(3 * (i + 1))
            continue
        if r.status_code == 200:
            return r
        print(f"   ! HTTP {r.status_code} dari {urlparse(url).netloc}", flush=True)
        print("   ! isi respon:", r.text[:150].replace("\n", " "), flush=True)
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
    # API Wikipedia cuma ngasih teks LENGKAP buat 1 artikel per request,
    # jadi ambil 1 artikel acak per request.
    print(f"[1/2] Ngambil {count} artikel acak dari Wikipedia ({lang})...", flush=True)
    total = added = 0
    for i in range(1, count + 1):
        r = get(f"https://{lang}.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "generator": "random",
            "grnnamespace": 0, "grnlimit": 1, "prop": "extracts",
            "explaintext": 1, "exlimit": 1})
        if r is None:
            continue
        try:
            pages = r.json().get("query", {}).get("pages", {})
        except ValueError:
            print("   ! respon bukan JSON:", r.text[:150].replace("\n", " "), flush=True)
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


def web(sources_file, seen):
    urls = []
    if os.path.exists(sources_file):
        urls = [u for u in open(sources_file).read().split() if u.startswith("http")]
    print(f"[2/2] Sumber lain dari {sources_file}: {len(urls)} URL", flush=True)
    total, robots = 0, {}
    for url in urls:
        host = urlparse(url)
        base = f"{host.scheme}://{host.netloc}"
        if base not in robots:
            rp = robotparser.RobotFileParser(base + "/robots.txt")
            try:
                rp.read()
            except Exception:
                rp = None
            robots[base] = rp
        rp = robots[base]
        if rp is None or not rp.can_fetch(UA["User-Agent"], url):
            print("  skip (robots.txt gak ngizinin):", url, flush=True)
            continue
        print("  ambil:", url, flush=True)
        r = get(url)
        if r is None:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        text = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        n = save_doc(text, seen)
        total += n
        print(f"  +{n} karakter" if n else "  (kosong/duplikat, dilewati)", flush=True)
        time.sleep(DELAY)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", type=int, default=100, help="jumlah artikel acak")
    ap.add_argument("--lang", default="id")
    ap.add_argument("--sources", default="sources.txt")
    a = ap.parse_args()
    print("=== Coops-AI Crawler ===", flush=True)
    seen = load_seen()
    print(f"Dokumen yang udah pernah diambil: {len(seen)}", flush=True)
    n = wikipedia(a.lang, a.wiki, seen) + web(a.sources, seen)
    size = os.path.getsize(CORPUS) if os.path.exists(CORPUS) else 0
    print(f"Selesai! +{n} karakter baru | corpus sekarang {size/1e6:.2f} MB", flush=True)
