import argparse, hashlib, os, time, requests
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
