"""Tokenizer BPE buatan sendiri (dilatih dari corpus lu, bukan punya orang lain)."""
import json, os, re
from collections import Counter, defaultdict
from common import CHARS

PRE = re.compile(r" ?\w+| ?[^\w\s]+|\s+")
SPECIALS = ["<|pengguna|>", "<|coops|>", "<|akhir|>"]
SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")
TOK_PATH = os.path.join("data", "tokenizer.json")
NB = len(CHARS)
CHAR2ID = {c: i for i, c in enumerate(CHARS)}


class BPE:
    def __init__(self, merges):
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.id2str = list(CHARS)
        for a, b in self.merges:
            self.id2str.append(self.id2str[a] + self.id2str[b])
        self.specials = {s: len(self.id2str) + i for i, s in enumerate(SPECIALS)}
        self.id2str += SPECIALS
        self.vocab_size = len(self.id2str)
        self.cache = {}

    # ---------- training ----------
    @classmethod
    def train(cls, text, vocab_size=4096, log=print):
        n_merges = vocab_size - NB - len(SPECIALS)
        words = Counter(PRE.findall(text))
        seqs, freqs = [], []
        for w, f in words.items():
            s = [CHAR2ID[c] for c in w if c in CHAR2ID]
            if len(s) > 0:
                seqs.append(s)
                freqs.append(f)
        log(f"  {len(seqs)} kata unik, mulai {n_merges} merge...")
        pair_counts = Counter()
        pair_words = defaultdict(set)
        for i, s in enumerate(seqs):
            f = freqs[i]
            for p in zip(s, s[1:]):
                pair_counts[p] += f
                pair_words[p].add(i)
        merges = []
        for m in range(n_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            if pair_counts[best] < 2:
                break
            a, b = best
            new = NB + len(merges)
            merges.append(best)
            for i in list(pair_words[best]):
                s, f = seqs[i], freqs[i]
                if len(s) < 2:
                    continue
                for p in zip(s, s[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                out, j = [], 0
                while j < len(s):
                    if j < len(s) - 1 and s[j] == a and s[j + 1] == b:
                        out.append(new)
                        j += 2
                    else:
                        out.append(s[j])
                        j += 1
                seqs[i] = out
                for p in zip(out, out[1:]):
                    pair_counts[p] += f
                    pair_words[p].add(i)
            pair_words.pop(best, None)
            pair_counts.pop(best, None)
            if (m + 1) % 500 == 0:
                log(f"  merge {m+1}/{n_merges}")
        return cls(merges)

    # ---------- encode / decode ----------
    def _word(self, w):
        ids = self.cache.get(w)
        if ids is not None:
            return ids
        ids = [CHAR2ID[c] for c in w if c in CHAR2ID]
        while len(ids) > 1:
            best, best_rank = None, None
            for p in zip(ids, ids[1:]):
                r = self.rank.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            new, out, j = NB + best_rank, [], 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == best[0] and ids[j + 1] == best[1]:
                    out.append(new)
                    j += 2
                else:
                    out.append(ids[j])
                    j += 1
            ids = out
        self.cache[w] = ids
        return ids

    def encode(self, text):
        out = []
        for part in SPECIAL_RE.split(text):
            if part in self.specials:
                out.append(self.specials[part])
            elif part:
                for w in PRE.findall(part):
                    out.extend(self._word(w))
        return out

    def decode(self, ids):
        return "".join(self.id2str[i] for i in ids)

    # ---------- simpan / muat ----------
    def save(self, path=TOK_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"merges": self.merges}, f)

    @classmethod
    def load(cls, path=TOK_PATH):
        if not os.path.exists(path):
            raise SystemExit("Tokenizer belum ada. Jalanin dulu: python prep.py")
        return cls(json.load(open(path))["merges"])
