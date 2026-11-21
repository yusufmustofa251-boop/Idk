import argparse, hashlib, os, time, requests
from urllib import robotparser
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from common import CORPUS, SEEN, clean

UA = {"User-Agent": "LearnerBot/0.1 (personal learning project)"}
MIN_LEN = 400
DELAY = 1.5


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


def wikipedia(lang, batches, seen):
    total = 0
    for _ in range(batches):
        r = requests.get(
            f"https://{lang}.wikipedia.org/w/api.php",
            params={"action": "query", "format": "json", "generator": "random",
                    "grnnamespace": 0, "grnlimit": 20, "prop": "extracts",
                    "explaintext": 1, "exlimit": "max"},
            headers=UA, timeout=30)
        for page in r.json().get("query", {}).get("pages", {}).values():
            total += save_doc(page.get("extract", ""), seen)
        time.sleep(DELAY)
    return total


def web(sources_file, seen):
    total = 0
    if not os.path.exists(sources_file):
        return 0
    robots = {}
    for url in open(sources_file).read().split():
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
            print("skip (robots):", url)
            continue
        try:
            html = requests.get(url, headers=UA, timeout=30).text
        except Exception as e:
            print("gagal:", url, e)
            continue
        soup = BeautifulSoup(html, "html.parser")
        text = "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
        total += save_doc(text, seen)
        time.sleep(DELAY)
    return total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", type=int, default=10, help="jumlah batch (x20 artikel)")
    ap.add_argument("--lang", default="id")
    ap.add_argument("--sources", default="sources.txt")
    a = ap.parse_args()
    seen = load_seen()
    n = wikipedia(a.lang, a.wiki, seen) + web(a.sources, seen)
    size = os.path.getsize(CORPUS) if os.path.exists(CORPUS) else 0
    print(f"+{n} karakter baru | corpus sekarang {size/1e6:.2f} MB")
