"""Tahap 3: bikin data tanya-jawab dari corpus lu sendiri (tanpa model/dataset orang lain).
Pola: kalimat pertama artikel "X adalah ..." jadi jawaban buat "Apa itu X?"."""
import random, re
import numpy as np
from common import CORPUS
from bpe import BPE

random.seed(42)
tok = BPE.load()
print("=== Coops-AI Chat Data ===", flush=True)

SAPAAN = [
    (["Halo", "Hai", "Halo!", "Hai Coops", "Hei"], "Halo! Aku Coops-AI. Ada yang bisa aku bantu?"),
    (["Selamat pagi", "Pagi"], "Selamat pagi! Ada yang bisa aku bantu?"),
    (["Selamat malam", "Malam"], "Selamat malam! Mau ngobrol apa hari ini?"),
    (["Apa kabar?", "Gimana kabarmu?"], "Baik! Kamu sendiri gimana?"),
    (["Siapa kamu?", "Kamu siapa?", "Kenalin diri kamu"],
     "Aku Coops-AI, model bahasa kecil yang dilatih dari nol di Codespaces."),
    (["Kamu bisa apa?", "Apa yang bisa kamu lakukan?"],
     "Aku bisa nyoba ngejelasin hal-hal umum. Tapi aku model kecil, jadi jawabanku belum tentu benar. Coba tanya 'Apa itu ...'."),
    (["Terima kasih", "Makasih", "Thanks"], "Sama-sama!"),
    (["Dadah", "Sampai jumpa", "Bye"], "Sampai jumpa! Semoga harimu menyenangkan."),
]


def conv(q, a):
    return f"<|pengguna|>{q}<|coops|>{a}<|akhir|>"


text = open(CORPUS, encoding="utf-8", errors="ignore").read()
docs = text.split("\n\n")
print(f"Dokumen di corpus: {len(docs)}", flush=True)

DEF = re.compile(r"^([^()\n.,;:]{2,50}?) (adalah|merupakan|ialah) (.+)$", re.S)
convs, seen_subj = [], set()
for d in docs:
    d = d.strip().replace("\n", " ")
    if len(d) < 200:
        continue
    sents = re.split(r"(?<=[.!?])\s+", d)
    m = DEF.match(sents[0])
    if not m:
        continue
    subj = m.group(1).strip()
    if (not subj or not subj[0].isupper() or len(subj.split()) > 6
            or subj in seen_subj or subj.lower().split()[0] in ("ini", "itu", "dia", "ia", "yang")):
        continue
    ans = ""
    for s in sents[:4]:
        if ans and len(ans) + len(s) > 450:
            break
        ans += (" " if ans else "") + s
    if not (40 <= len(ans) <= 600) or ans[-1] not in ".!?":
        continue
    seen_subj.add(subj)
    q = random.choice([f"Apa itu {subj}?", f"Jelaskan tentang {subj}.", f"{subj} itu apa?"])
    convs.append(conv(q, ans))
n_def = len(convs)

reps = 15
for qs, ans in SAPAAN:
    for i in range(reps):
        convs.append(conv(qs[i % len(qs)], ans))

if n_def < 300:
    print(f"! Cuma {n_def} pasangan tanya-jawab. Tambah corpus dulu biar chatbot-nya lebih bagus.", flush=True)
random.shuffle(convs)
ids = []
for c in convs:
    ids.extend(tok.encode(c))
arr = np.array(ids, dtype=np.uint16)
np.save("data/chat_tokens.npy", arr)
print(f"Selesai! {n_def} tanya-jawab dari corpus + {len(SAPAAN)*reps} sapaan | {len(arr)/1e6:.2f} juta token", flush=True)
print("Contoh:", convs[0][:200], flush=True)
