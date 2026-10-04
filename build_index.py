#!/usr/bin/env python3
"""
build_index.py - proof-of-concept chunking + embedding over data/docs.jsonl (output of garante_scraper.py parse).

Stages:
  chunk   select the newest N clean documents, collapse near-duplicate templates, drop corpus-wide boilerplate,
          split into ~400-token chunks                                   -> data/poc/chunks.jsonl, duplicates.json
  embed   embed chunks (--model jasper | bge-m3)                          -> data/poc/embeddings[_<model>].npy
  bm25    keyword index over the chunks (Italian stemming)                -> data/poc/bm25.npz, bm25_vocab.json
  search  hybrid search (embeddings + BM25, fused by rank) grouped by document

  python build_index.py chunk --n 3000
  python build_index.py embed --model jasper
  python build_index.py bm25
  python build_index.py search "videosorveglianza sul luogo di lavoro"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

DATA = Path("data")
OUT = DATA / "poc"
MODEL = "BAAI/bge-m3"    # its tokenizer sets the chunk sizes; the chunks are shared by all embedding models
# Embedding models. Each one embeds the same chunks.jsonl into its own file, so they can be compared side by side.
MODELS = {
    "bge-m3": {"name": "BAAI/bge-m3", "file": "embeddings.npy", "query": {}, "doc": {}},
    # Bilingual (zh/en) distillation of Qwen3-Embedding-0.6B with token compression; 2048d.
    # Runs custom code from the repo (reviewed: plain PyTorch), pinned to the reviewed revision.
    "jasper": {"name": "infgrad/Jasper-Token-Compression-600M", "file": "embeddings_jasper.npy",
               "revision": "e1373068c35fba2a37568da4eb4105377e17eae9", "dtype": "bfloat16",  # float16 gives NaNs
               "query": {"prompt_name": "query", "compression_ratio": 0.5}, "doc": {"compression_ratio": 0.5}},
}
DEFAULT_MODEL = "jasper"
CHUNK_TOKENS = 400      # target body tokens per chunk (bge-m3 tokenizer), header excluded
OVERLAP_TOKENS = 60     # carried over from the previous chunk, in whole sentences
MIN_TAIL_TOKENS = 80    # a smaller final chunk is merged into the previous one
MIN_SECTION_TOKENS = 120  # a section change starts a new chunk once the current one is at least this big
MIN_BODY_CHARS = 300    # shorter bodies are stubs (PDF-only pages, redirects)
EMBED_SLICE = 2048     # chunks per saved embedding slice (resumable)
DUP_JACCARD = 0.7       # decisions whose 5-word shingles overlap this much are one template family
BOILERPLATE_MIN_DOCS = 25   # a paragraph repeated verbatim in this many distinct decisions is boilerplate
BOILERPLATE_MIN_CHARS = 60  # shorter paragraphs (headings like "DISPONE") are never treated as boilerplate

STUB = re.compile(r"Attendere prego|^Redirect$", re.M)
DOCWEB_HEADER = re.compile(r"^\[\s*doc\.?\s*web\s*n\.?\s*\d+\s*\]\s*", re.I)
# Section openers in Garante decisions; always start a new paragraph.
HEADING = re.compile(r"^(IL GARANTE|PREMESSO|RILEVAT[OA]|CONSIDERAT[OA]|OSSERVA|RITENUT[OA]|VIST[OAIE]\b|ESAMINAT[OA]|"
                     r"TUTTO CI[OÒ] PREMESSO|CI[OÒ] PREMESSO|PER QUESTI MOTIVI|DICHIARA|ORDINA|INGIUNGE|DISPONE|"
                     r"IL PRESIDENTE|IL RELATORE|IL SEGRETARIO GENERALE|Roma,? \d|"
                     r"OGGETTO|[0-9]{1,2}\.\s|[a-z]\)\s)", re.I)
SENT_END = re.compile(r"[.;:!?][\"»”]?$")
CONTINUATION = re.compile(r"^[,.;:)\]»”]")  # a line starting with punctuation continues the previous one
SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-ZÀÈÉÌÒÙ«“\"(\[0-9])")

# Sections of a decision. Openers are matched case-sensitively (they are upper case in the decisions);
# paragraphs without an opener inherit the current section.
SECTION_NAMES = {"sintesi": "Sintesi", "fatti": "Fatto", "motivazione": "Motivazione", "dispositivo": "Decisione",
                 "normativa": "Quadro normativo", "procedura": "Atti del procedimento", "testo": "Testo"}
# Search weights: the case's activity (sintesi, fatti) and the violation (motivazione, dispositivo) rank first.
SECTION_WEIGHTS = {"sintesi": 1.0, "fatti": 1.0, "motivazione": 1.0,
                   "dispositivo": 0.97, "normativa": 0.97, "testo": 0.97, "procedura": 0.9}
ROLES = {"sintesi": "Attività", "fatti": "Attività", "motivazione": "Violazione", "dispositivo": "Violazione"}
DROP_SECTIONS = {"preambolo", "firma"}  # attendance list ("in presenza del prof. ...") and signatures
OPEN_PREAMBOLO = re.compile(r"^IL GARANTE PER LA PROTEZIONE DEI DATI PERSONALI\b")
OPEN_FIRMA = re.compile(r"^(IL PRESIDENTE|IL RELATORE|IL SEGRETARIO GENERALE|IL VICE ?PRESIDENTE|Roma,? \d)")
OPEN_PREMESSO_GARANTE = re.compile(r"^(TUTTO )?CI[OÒ] PREMESSO,? IL GARANTE")
OPEN_DISPOSITIVO = re.compile(r"^(PER QUESTI MOTIVI|DICHIARA|ORDINA|INGIUNGE|DISPONE|PRESCRIVE|VIETA|ACCOGLIE)\b")
OPEN_FATTI = re.compile(r"^(PREMESSO|RILEVAT[OA]|ESAMINAT[OA])\b")
OPEN_MOTIVAZIONE = re.compile(r"^(CONSIDERAT[OA]|RITENUT[OA]|OSSERVAT[OA]|OSSERVA|TENUTO CONTO)\b")
OPEN_VISTO = re.compile(r"^VIST[OAIE]\b")
# GDPR-era decisions number their sections: "1. Introduzione", "3. VALUTAZIONI DI ORDINE GIURIDICO".
# A heading is a short numbered line without final punctuation; its words decide the section (first match wins).
NUMBERED_HEADING = re.compile(r"^\d{1,2}(\.\d{1,2})*\.?\s+\S[^.;:,]{2,80}$")
HEADING_SECTIONS = [
    ("dispositivo", re.compile(r"ordinanza|applicazione della sanzion|quantificazion|misure correttive|provvedimenti correttivi|corrective", re.I)),
    ("motivazione", re.compile(r"esit|valutazion|contestazion|violazioni|illiceit|illegittimit|conclusion|"
                               r"osservazion|difes|memori|argomentazion|quadro (giuridico|normativo)|normativa|"
                               r"base giuridic|basi giuridic|titolarit|profili|risultanz|ambito di applicazione|"
                               r"findings|^\S+\s+sul(l[a'’o]|l?e|\s)", re.I)),
    ("fatti", re.compile(r"introduzion|premessa|reclamo|istruttori|segnalazion|richiest|ispett|accertament|"
                         r"notifica|violazione d|misure|ricostruzion|istanza|il caso|il fatto|trattamento|"
                         r"fact-finding", re.I)),
]
DROP_PARAGRAPH = re.compile(r"^Registro dei provvedimenti\b")
# A VISTO paragraph citing a law states what the case is about (e.g. art. 144-bis on intimate images), so it is
# legal framework rather than procedure. Generic citations (the GDPR, the Codice) are removed as boilerplate.
VISTO_NORMA = re.compile(r"^VIST[OAIE]\s+(altresì\s+)?(l[’'a]\s*|il\s+|i\s+|gli\s+|le\s+)?"
                         r"((medesim|citat|predett)[oaie]\s+)?(art|legge|d\.?\s?l|decret|"
                         r"codice|regolament|direttiva|d\.p\.r|provvedimento|parere|linee guida)", re.I)
# A VISTO paragraph that reports what a party did or claimed is part of the facts, not procedure.
VISTO_FACTS = re.compile(r"con (la|il|le|i) qual[ei]|lamenta|ha chiesto|chiede|ha rappresentato|ha dichiarato|"
                         r"ha affermato|ha precisato|ha sostenuto|ha segnalato|ha contestato|deduce", re.I)


# ----------------------------------------------------------------------------- cleaning
def label_sections(paras: list[str], title: str) -> list[tuple[str, str]]:
    """(section, paragraph) pairs; text before the preamble is the editorial summary when the doc has one."""
    has_preamble = any(OPEN_PREAMBOLO.match(p) for p in paras)
    bare_title = re.sub(r"\s*\[\d+\]\s*$", "", title).strip()
    section = "sintesi" if has_preamble else "testo"
    out = []
    for p in paras:
        if DROP_PARAGRAPH.match(p):
            continue
        if NUMBERED_HEADING.match(p) and section not in ("preambolo", "firma"):
            section = next((s for s, rx in HEADING_SECTIONS if rx.search(p)), section)
        elif OPEN_PREAMBOLO.match(p):
            section = "preambolo"
        elif OPEN_FIRMA.match(p):
            section = "firma"
        elif OPEN_PREMESSO_GARANTE.match(p):
            section = "motivazione" if "OSSERVA" in p[:60] else "dispositivo"
        elif OPEN_DISPOSITIVO.match(p):
            section = "dispositivo"
        elif OPEN_MOTIVAZIONE.match(p):
            section = "motivazione" if section != "dispositivo" else section
        elif OPEN_FATTI.match(p):
            section = "fatti" if section not in ("motivazione", "dispositivo") else section
        elif OPEN_VISTO.match(p):
            section = ("fatti" if VISTO_FACTS.search(p) else "normativa" if VISTO_NORMA.match(p) else "procedura")
        elif section in ("preambolo", "firma"):
            section = "testo"  # unlabelled text after the preamble or signatures
        if section == "sintesi" and bare_title and p.startswith(bare_title):
            p = p[len(bare_title):].strip(" -–:")  # the summary often repeats the title
            if not p or DROP_PARAGRAPH.match(p):
                continue
        out.append((section, p))
    return [(s, p) for s, p in out if s not in DROP_SECTIONS]


def paragraphs(body: str) -> list[str]:
    """Rejoin lines broken at inline HTML elements into real paragraphs."""
    body = DOCWEB_HEADER.sub("", body.strip()).replace("´", "'").replace("`", "'")
    paras, cur = [], []
    for line in body.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        if cur and not CONTINUATION.match(line) and (HEADING.match(line) or SENT_END.search(cur[-1])
                                                     or NUMBERED_HEADING.match(cur[-1])):
            paras.append(" ".join(cur))
            cur = []
        cur.append(line)
    if cur:
        paras.append(" ".join(cur))
    # tidy spacing left by the line joins: "( testo )", "parola ,"
    return [re.sub(r"\s+([,.;:)\]])", r"\1", re.sub(r"([(\[])\s+", r"\1", p)) for p in paras]


def select_docs(n: int) -> tuple[list[dict], Counter]:
    recs = [json.loads(line) for line in (DATA / "docs.jsonl").open(encoding="utf-8")]
    recs.sort(key=lambda r: (r["date"] or "", r["docweb_id"]), reverse=True)
    dropped, seen, out = Counter(), set(), []
    for r in recs:
        if len(out) == n:
            break
        if r["body_chars"] < MIN_BODY_CHARS:
            dropped["short_or_empty"] += 1
        elif STUB.search(r["body"]):
            dropped["redirect_stub"] += 1
        elif r["body_sha256"] in seen:
            dropped["duplicate_body"] += 1
        else:
            seen.add(r["body_sha256"])
            out.append(r)
    return out, dropped


# ----------------------------------------------------------------------------- templates + boilerplate
def _shingles(body: str) -> set[int]:
    text = re.sub(r"\d+", "0", body.lower())
    text = re.sub(r"\bx{2,}\b|\[omissis\]", "x", text)  # anonymised names
    words = re.findall(r"\w+", text)
    return {hash(" ".join(words[i:i + 5])) for i in range(max(1, len(words) - 4))}


def near_duplicate_clusters(docs: list[dict]) -> list[list[int]]:
    """Groups of near-identical decisions (template families such as the "revenge porn" ratifications), found
    with MinHash + LSH and confirmed by exact Jaccard >= DUP_JACCARD. Indices into `docs`, each group in input
    order (so with docs sorted newest first, the first index is the newest member and becomes the representative)."""
    import numpy as np
    sets = [_shingles(d["body"]) for d in docs]
    rng = np.random.default_rng(0)
    prime = np.uint64((1 << 61) - 1)
    a = rng.integers(1, int(prime), 128, dtype=np.uint64)
    b = rng.integers(0, int(prime), 128, dtype=np.uint64)
    sig = np.zeros((len(sets), 128), dtype=np.uint64)
    for i, sh in enumerate(sets):
        h = np.fromiter((x & 0xFFFFFFFFFFFF for x in sh), dtype=np.uint64)  # 48 bits: no overflow in a*h
        sig[i] = ((np.outer(h, a) + b) % prime).min(axis=0)
    parent = list(range(len(sets)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    seen = set()
    for band in range(32):  # 32 bands x 4 rows: catches pairs with Jaccard ~0.5 and above
        buckets: dict[bytes, list[int]] = {}
        for i in range(len(sets)):
            buckets.setdefault(sig[i, band * 4:(band + 1) * 4].tobytes(), []).append(i)
        for members in buckets.values():
            for x in members[1:]:
                pair = (members[0], x)
                if pair in seen or find(pair[0]) == find(pair[1]):
                    continue
                seen.add(pair)
                inter = len(sets[pair[0]] & sets[pair[1]])
                if inter / (len(sets[pair[0]]) + len(sets[pair[1]]) - inter) >= DUP_JACCARD:
                    parent[find(pair[0])] = find(pair[1])
    groups: dict[int, list[int]] = {}
    for i in range(len(sets)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _is_boilerplate(p: str, keys: frozenset[str] | set[str]) -> bool:
    # numbered headings repeat across decisions but give each chunk its context, so they always stay
    return len(p) >= BOILERPLATE_MIN_CHARS and not NUMBERED_HEADING.match(p) and _para_key(p) in keys


def _para_key(p: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", p.lower())).strip()


def boilerplate_keys(docs: list[dict]) -> set[str]:
    """Paragraphs repeated in at least BOILERPLATE_MIN_DOCS decisions ("VISTO il Regolamento (UE) 2016/679 ...",
    "VISTA la documentazione in atti", the notice on how to appeal). Count over template representatives only,
    so a family of 700 copies does not turn its own subject matter into boilerplate."""
    df = Counter()
    for doc in docs:
        df.update({_para_key(p) for _, p in label_sections(paragraphs(doc["body"]), doc["title"])
                   if len(p) >= BOILERPLATE_MIN_CHARS and not NUMBERED_HEADING.match(p)})
    return {k for k, n in df.items() if n >= BOILERPLATE_MIN_DOCS}


# ----------------------------------------------------------------------------- chunking
def header(doc: dict, section: str) -> str:
    parts = [doc["title"], f"Data: {doc['date']}"]
    if doc["tipologia"]:
        parts.append("Tipologia: " + ", ".join(doc["tipologia"]))
    if doc["argomenti"]:
        parts.append("Argomenti: " + ", ".join(doc["argomenti"]))
    parts.append("Sezione: " + SECTION_NAMES[section])
    return " | ".join(parts)


def chunk_doc(doc: dict, ntok, boilerplate: frozenset[str] = frozenset()) -> list[tuple[str, dict[str, float]]]:
    """(text, section token shares) per chunk. A section change starts a new chunk once the current one
    has MIN_SECTION_TOKENS, so most chunks belong to a single section; overlap never crosses sections."""
    # Units are sentences grouped by paragraph; a paragraph longer than the budget is split into sentences.
    units: list[tuple[str, int, bool, str]] = []  # (text, tokens, starts_paragraph, section)
    for section, p in label_sections(paragraphs(doc["body"]), doc["title"]):
        if _is_boilerplate(p, boilerplate):
            continue
        sents = [p] if ntok(p) <= CHUNK_TOKENS else SENT_SPLIT.split(p)
        for i, s in enumerate(sents):
            units.append((s, ntok(s), i == 0, section))
    size = lambda c: sum(u[1] for u in c)
    chunks: list[list[tuple[str, int, bool, str]]] = []
    cur: list[tuple[str, int, bool, str]] = []
    for u in units:
        new_section = bool(cur) and u[3] != cur[-1][3] and size(cur) >= MIN_SECTION_TOKENS
        if cur and (new_section or size(cur) + u[1] > CHUNK_TOKENS):
            chunks.append(cur)
            tail, n = [], 0  # overlap: trailing units of the previous chunk, up to OVERLAP_TOKENS
            for prev in reversed(cur):
                if new_section or prev[3] != u[3] or n + prev[1] > OVERLAP_TOKENS:
                    break
                tail.insert(0, prev)
                n += prev[1]
            cur = tail
        cur.append(u)
    if cur:
        if chunks and size(cur) < MIN_TAIL_TOKENS:
            chunks[-1].extend(x for x in cur if x not in chunks[-1])
        else:
            chunks.append(cur)
    out = []
    for c in chunks:
        text, shares = "", Counter()
        for s, t, starts_para, section in c:
            text += ("\n" if starts_para and text else " " if text else "") + s
            shares[section] += t
        total = sum(shares.values())
        out.append((text, {k: round(v / total, 3) for k, v in shares.most_common()}))
    return out


def cmd_chunk(args) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    docs, dropped = select_docs(args.n)
    if len(docs) < args.n:
        print(f"[warn] only {len(docs)} usable documents")
    OUT.mkdir(parents=True, exist_ok=True)
    # Template families: index the newest member, list the others with it (shown as "similar decisions").
    clusters = near_duplicate_clusters(docs)
    reps = [docs[g[0]] for g in clusters]
    similar = {docs[g[0]]["docweb_id"]: [{k: docs[i][k] for k in ("docweb_id", "date", "title", "url")}
                                         for i in g[1:]] for g in clusters if len(g) > 1}
    (OUT / "duplicates.json").write_text(json.dumps(similar, ensure_ascii=False), encoding="utf-8")
    boilerplate = frozenset(boilerplate_keys(reps))
    n_chunks, sizes, sections = 0, [], Counter()
    with (OUT / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for doc in reps:
            n_similar = len(similar.get(doc["docweb_id"], []))
            for i, (text, shares) in enumerate(chunk_doc(doc, ntok, boilerplate)):
                section = next(iter(shares))  # majority section by tokens
                rec = {"chunk_id": f"{doc['docweb_id']}_{i:03d}", "docweb_id": doc["docweb_id"], "chunk_index": i,
                       "title": doc["title"], "date": doc["date"], "url": doc["url"],
                       "tipologia": doc["tipologia"], "argomenti": doc["argomenti"],
                       "section": section, "section_shares": shares, "similar_count": n_similar,
                       "weight": round(sum(SECTION_WEIGHTS[k] * v for k, v in shares.items()), 4),
                       "text": text, "embed_text": f"{header(doc, section)}\n\n{text}"}
                rec["n_tokens"] = ntok(rec["embed_text"])
                sizes.append(rec["n_tokens"])
                sections[section] += 1
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1
    sizes.sort()
    meta = {"documents": len(docs), "date_range": [docs[-1]["date"], docs[0]["date"]], "dropped": dropped,
            "indexed_documents": len(reps), "template_families": sum(1 for g in clusters if len(g) > 1),
            "largest_families": sorted((len(g) for g in clusters), reverse=True)[:5],
            "boilerplate_paragraphs": len(boilerplate),
            "chunks": n_chunks, "chunks_per_doc": round(n_chunks / len(reps), 2),
            "chunks_by_section": dict(sections.most_common()),
            "embed_tokens_p10_p50_p90_max": [sizes[int(p * (len(sizes) - 1))] for p in (.1, .5, .9, 1)],
            "chunk_tokens": CHUNK_TOKENS, "overlap_tokens": OVERLAP_TOKENS, "model": MODEL,
            "source_sha256": hashlib.sha256((DATA / "docs.jsonl").read_bytes()).hexdigest()}
    (OUT / "chunks_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in meta.items():
        print(f"{k}: {v}")


# ----------------------------------------------------------------------------- embedding + search
def load_model(key: str = DEFAULT_MODEL):
    import torch
    from sentence_transformers import SentenceTransformer
    spec = MODELS[key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": getattr(torch, spec.get("dtype", "float16"))} if device == "cuda" else {}
    if "revision" not in spec:
        model = SentenceTransformer(spec["name"], device=device, model_kwargs=model_kwargs)
    else:
        model_kwargs.update(trust_remote_code=True, attn_implementation="sdpa")
        model = SentenceTransformer(spec["name"], revision=spec["revision"], device=device, trust_remote_code=True,
                                    model_kwargs=model_kwargs, tokenizer_kwargs={"padding_side": "left"})
    model.max_seq_length = 1024
    return model


def encode_queries(model, key: str, texts: list[str]):
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, **MODELS[key]["query"])


def encode_docs(model, key: str, texts: list[str], batch_size: int = 16):
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                        **MODELS[key]["doc"])


def embeddings_path(key: str = DEFAULT_MODEL) -> Path:
    return OUT / MODELS[key]["file"]


# ----------------------------------------------------------------------------- BM25 + hybrid
BM25_K1, BM25_B = 1.2, 0.75
RRF_K = 60              # reciprocal rank fusion constant
HYBRID_ALPHA = 0.3      # share of BM25 in the fused ranking (0 = embeddings only, 1 = BM25 only)
STOPWORDS_IT = set("""a ad al alla alle allo agli ai all anche c che chi ci cio ciò come con contro cui d da dal
dalla dalle dallo dagli dai dall degli dei del della delle dello dell di dove e ed è era essere gli ha hanno i il in
io la le lo l loro ma mi ne nei nel nella nelle nello negli nell non o per però più piu quale quali quando quanto
quello questa queste questi questo se si sia sono su sul sulla sulle sullo sugli sui sull tra fra un una uno
data tipologia argomenti sezione""".split())  # the last four are the chunk header's own labels
_WORD = re.compile(r"[a-zà-öø-ÿ0-9]+")


class Analyzer:
    """Lower-case, split on non-letters (so "l'istanza" -> "istanza"), drop stopwords, Italian Snowball stems."""
    def __init__(self) -> None:
        import snowballstemmer
        self._stem = snowballstemmer.stemmer("italian").stemWord
        self._cache: dict[str, str] = {}

    def __call__(self, text: str) -> list[str]:
        out = []
        for w in _WORD.findall(text.lower()):
            if w in STOPWORDS_IT or len(w) < 2:
                continue
            s = self._cache.get(w)
            if s is None:
                s = self._cache[w] = self._stem(w)
            out.append(s)
        return out


def _chunks_key() -> str:
    return hashlib.sha256((OUT / "chunks.jsonl").read_bytes()).hexdigest()[:12]


def cmd_bm25(args) -> None:
    """Precompute BM25 term weights per chunk, so a query is a sum of a few sparse columns."""
    import numpy as np
    from scipy import sparse
    analyze = Analyzer()
    vocab: dict[str, int] = {}
    rows, cols, vals, lengths = [], [], [], []
    for i, line in enumerate((OUT / "chunks.jsonl").open(encoding="utf-8")):
        tf = Counter(analyze(json.loads(line)["embed_text"]))
        lengths.append(sum(tf.values()))
        for term, n in tf.items():
            rows.append(i)
            cols.append(vocab.setdefault(term, len(vocab)))
            vals.append(n)
    n_docs = len(lengths)
    m = sparse.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(n_docs, len(vocab)))
    lengths = np.array(lengths, dtype=np.float32)
    df = np.bincount(m.indices, minlength=len(vocab))
    idf = np.log1p((n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    norm = BM25_K1 * (1 - BM25_B + BM25_B * lengths / lengths.mean())
    row_of = np.repeat(np.arange(n_docs), np.diff(m.indptr))
    m.data = m.data * (BM25_K1 + 1) / (m.data + norm[row_of]) * idf[m.indices]
    sparse.save_npz(OUT / "bm25.npz", m.tocsc())
    (OUT / "bm25_vocab.json").write_text(json.dumps({"chunks_key": _chunks_key(), "vocab": vocab}, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"[bm25] {n_docs} chunks, {len(vocab):,} stems, {m.nnz:,} postings -> {OUT / 'bm25.npz'}")


class BM25:
    def __init__(self) -> None:
        from scipy import sparse
        meta = json.loads((OUT / "bm25_vocab.json").read_text(encoding="utf-8"))
        if meta["chunks_key"] != _chunks_key():
            raise RuntimeError("bm25 index is out of date: run `python build_index.py bm25`")
        self.vocab = meta["vocab"]
        self.matrix = sparse.load_npz(OUT / "bm25.npz").tocsc()
        self.analyze = Analyzer()

    def scores(self, query: str):
        import numpy as np
        cols = [self.vocab[t] for t in set(self.analyze(query)) if t in self.vocab]
        if not cols:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        return np.asarray(self.matrix[:, cols].sum(axis=1)).ravel()


def fuse(dense, lexical, alpha: float):
    """Weighted reciprocal rank fusion of two chunk score arrays; chunks without any query term get no BM25 share.
    alpha=0 returns the dense scores unchanged (cosine), alpha=1 the BM25 scores."""
    import numpy as np
    if alpha <= 0:
        return dense
    if alpha >= 1:
        return lexical
    n = len(dense)
    rank_d = np.empty(n, dtype=np.float32)
    rank_d[np.argsort(-dense)] = np.arange(n)
    rank_l = np.empty(n, dtype=np.float32)
    rank_l[np.argsort(-lexical)] = np.arange(n)
    return (1 - alpha) / (RRF_K + rank_d) + alpha * (lexical > 0) / (RRF_K + rank_l)


def score_chunks(q_vec, query: str, emb, bm25: BM25 | None, weights=None, alpha: float = HYBRID_ALPHA):
    """Chunk scores for a query: section-weighted cosine, fused with section-weighted BM25 when alpha > 0."""
    dense = emb @ q_vec
    if weights is not None:
        dense = dense * weights
    if bm25 is None or alpha <= 0:
        return dense
    lexical = bm25.scores(query)
    if weights is not None:
        lexical = lexical * weights
    return fuse(dense, lexical, alpha)


def cmd_embed(args) -> None:
    import numpy as np
    texts = [json.loads(line)["embed_text"] for line in (OUT / "chunks.jsonl").open(encoding="utf-8")]
    # Slices are saved as they finish, so an interrupted run resumes; they are keyed to this chunks.jsonl.
    key = hashlib.sha256((OUT / "chunks.jsonl").read_bytes()).hexdigest()[:12]
    parts = OUT / f"embed_parts_{args.model}_{key}"
    parts.mkdir(exist_ok=True)
    model = load_model(args.model)
    t0 = time.perf_counter()
    for start in range(0, len(texts), EMBED_SLICE):
        path = parts / f"{start:07d}.npy"
        if path.exists():
            continue
        batch = texts[start:start + EMBED_SLICE]
        # Sort by length so batches have similar padding; restore the original order afterwards.
        order = sorted(range(len(batch)), key=lambda i: len(batch[i]))
        emb = encode_docs(model, args.model, [batch[i] for i in order], batch_size=args.batch_size)
        out = np.empty_like(emb)
        out[order] = emb
        np.save(path, out.astype(np.float16))
        print(f"[embed {args.model}] {min(start + EMBED_SLICE, len(texts))}/{len(texts)} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    emb = np.concatenate([np.load(p) for p in sorted(parts.glob("*.npy"))])
    assert len(emb) == len(texts), (len(emb), len(texts))
    np.save(embeddings_path(args.model), emb)
    print(f"[embed {args.model}] {emb.shape} -> {embeddings_path(args.model)}")


def rank_documents(scores, chunks: list[dict], k: int, mask=None) -> dict[int, dict[str, tuple[float, int]]]:
    """Documents ranked by their best chunk score -> {docweb_id: {"top"|"Attività"|"Violazione": (score, chunk idx)}}.
    Besides the top chunk, each result keeps the best passage on the case's activity (sintesi/fatti) and on the
    violation (motivazione/dispositivo), when the document has them. `mask` (bool array) excludes chunks."""
    import numpy as np
    order = np.argsort(-scores)
    if mask is not None:
        order = order[mask[order]]
    best: dict[int, dict[str, tuple[float, int]]] = {}
    for i in order:
        c = chunks[i]
        d = c["docweb_id"]
        if d not in best:
            if len(best) == k:
                continue
            best[d] = {"top": (float(scores[i]), int(i))}
        role = ROLES.get(c.get("section", "testo"))
        if role and role not in best[d]:
            best[d][role] = (float(scores[i]), int(i))
    return best


def cmd_search(args) -> None:
    import numpy as np
    chunks = [json.loads(line) for line in (OUT / "chunks.jsonl").open(encoding="utf-8")]
    emb = np.load(embeddings_path(args.model)).astype(np.float32)
    q = encode_queries(load_model(args.model), args.model, [args.query])[0]
    weights = None if args.no_weights else np.array([c.get("weight", 1.0) for c in chunks], dtype=np.float32)
    scores = score_chunks(q, args.query, emb, BM25() if args.alpha > 0 else None, weights, args.alpha)
    for rank, (d, hits) in enumerate(rank_documents(scores, chunks, args.k).items(), 1):
        s, i = hits["top"]
        c = chunks[i]
        print(f"{rank:2d}. {s:.3f}  [{c['date']}] {c['title'][:100]}\n    {c['url']}")
        shown = [(r, hits[r]) for r in ("Attività", "Violazione") if r in hits] or [(SECTION_NAMES.get(
            c.get("section", "testo"), "Testo"), hits["top"])]
        for role, (rs, ri) in shown:
            print(f"    {role} ({rs:.3f}): {chunks[ri]['text'][:260].replace(chr(10), ' ')}...")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("chunk")
    p.add_argument("--n", type=int, default=3000, help="number of newest documents to include")
    p = sub.add_parser("embed")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--no-weights", action="store_true", help="ignore section weights (plain cosine)")
    p.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    p.add_argument("--alpha", type=float, default=HYBRID_ALPHA, help="BM25 share in the fusion (0 = embeddings only)")
    sub.add_parser("bm25")
    args = ap.parse_args()
    {"chunk": cmd_chunk, "embed": cmd_embed, "bm25": cmd_bm25, "search": cmd_search}[args.cmd](args)


if __name__ == "__main__":
    main()
