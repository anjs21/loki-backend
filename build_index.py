#!/usr/bin/env python3
"""
build_index.py - proof-of-concept chunking + embedding over data/docs.jsonl (output of garante_scraper.py parse).

Stages:
  chunk   select the newest N clean documents, collapse near-duplicate templates, drop corpus-wide boilerplate
          and procedural formulas, label sections, split along sections and numbered sub-sections into chunks
          of up to ~400 tokens                                            -> data/poc/chunks.jsonl, duplicates.json
  embed   embed chunks (--model bidirlm, the default | jasper | bge-m3 | jina-v5-nano) -> data/poc/embeddings*.npy
  bm25    keyword index over the chunks (Italian stemming)                -> data/poc/bm25.npz, bm25_vocab.json
  search  hybrid search (embeddings + BM25, fused by rank) grouped by document

  python build_index.py chunk              # all documents (--n 3000: only the newest 3000)
  python build_index.py embed
  python build_index.py bm25
  python build_index.py search "videosorveglianza sul luogo di lavoro"
  python build_index.py search "conservazione eccessiva delle immagini" --sections violation
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
# Models with a "revision" run custom code from their repo: each was reviewed (plain PyTorch, no network or file
# access beyond loading their own weights) and is pinned to the reviewed revision.
MODELS = {
    "bge-m3": {"label": "BGE-M3", "name": "BAAI/bge-m3", "file": "embeddings.npy", "query": {}, "doc": {}},
    # Bilingual (zh/en) distillation of Qwen3-Embedding-0.6B with token compression; 2048d.
    "jasper": {"label": "Jasper 600M", "name": "infgrad/Jasper-Token-Compression-600M",
               "file": "embeddings_jasper.npy", "revision": "e1373068c35fba2a37568da4eb4105377e17eae9",
               "dtype": "bfloat16",  # float16 gives NaNs
               "model_kwargs": {"attn_implementation": "sdpa"}, "tokenizer_kwargs": {"padding_side": "left"},
               "query": {"prompt_name": "query", "compression_ratio": 0.5}, "doc": {"compression_ratio": 0.5}},
    # EuroBERT-210M with per-task LoRA adapters; 768d, last-token pooling. Licence CC-BY-NC-4.0 (non-commercial).
    "jina-v5-nano": {"label": "Jina v5 nano", "name": "jinaai/jina-embeddings-v5-text-nano",
                     "file": "embeddings_jina-v5-nano.npy", "revision": "8a7f00aac812071b69403df470f1038ec85f8925",
                     "dtype": "bfloat16", "licence": "CC-BY-NC-4.0",
                     "query": {"task": "retrieval", "prompt_name": "query"},
                     "doc": {"task": "retrieval", "prompt_name": "document"}},
    # Qwen3-0.6B made bidirectional (MNTP) + contrastive training; 1024d, mean pooling. Instruction on queries only.
    "bidirlm": {"label": "BidirLM 0.6B", "name": "BidirLM/BidirLM-0.6B-Embedding",
                "file": "embeddings_bidirlm.npy", "revision": "f6279763d6bd59ff963897965f29fed0b447ed3c",
                "dtype": "bfloat16",
                # Without flash-attn its attention packs the batch into one sequence and builds the full
                # (total tokens)^2 matrix, so cost per chunk grows with the batch: 1 is fastest on an 8 GB GPU
                # (25 chunks/s, vs 17.5 at 2, 10.7 at 4, 6.1 at 8; 16 runs out of memory).
                "batch_size": 1,
                "query": {"prompt": "Given a web search query, retrieve relevant passages that answer the query\n"},
                "doc": {}},
}
DEFAULT_MODEL = "bidirlm"
CHUNK_TOKENS = 400      # target body tokens per chunk (bge-m3 tokenizer), header excluded
OVERLAP_TOKENS = 60     # carried over from the previous chunk, in whole sentences
MIN_TAIL_TOKENS = 80    # a smaller final chunk is merged into the previous one
MIN_SECTION_TOKENS = 120  # a section change starts a new chunk once the current one is at least this big
MIN_BODY_CHARS = 300    # shorter bodies are stubs (PDF-only pages, redirects)
EMBED_SLICE = 2048     # chunks per saved embedding slice (resumable)
DUP_JACCARD = 0.7       # decisions whose 5-word shingles overlap this much are one template family
BOILERPLATE_MIN_DOCS = 25   # a paragraph repeated verbatim in this many distinct decisions is boilerplate
BOILERPLATE_MIN_CHARS = 60  # shorter paragraphs (headings like "DISPONE") are never treated as boilerplate
# Procedural formulas ("INGIUNGE ... di pagare la somma di euro ...", the register annotation, the appeal notice)
# vary in names and amounts, so they are matched by phrase: a 5-word phrase found in BOILERPLATE_MIN_DOCS decisions
# is procedural when no single topic tag covers more than FORMULA_MAX_TOPIC of them (wording shared within one topic,
# such as the Google delisting reasoning, is what those cases are about and stays). A sentence made of at least
# FORMULA_MIN_SHARE procedural phrases is dropped, unless it states which articles were violated.
FORMULA_MAX_TOPIC = 0.5
FORMULA_MIN_SHARE = 0.6
# Topic spread is measured over the tagged decisions that use a phrase (about half of the pre-2018 decisions have
# no tags, and they would make every phrase look spread); with fewer tagged decisions than this it cannot be judged.
FORMULA_MIN_TAGGED = 12

STUB = re.compile(r"Attendere prego|^Redirect$", re.M)
DOCWEB_HEADER = re.compile(r"^\[\s*doc\.?\s*web\s*n\.?\s*\d+\s*\]\s*", re.I)
# Section openers in Garante decisions; always start a new paragraph.
HEADING = re.compile(r"^(IL GARANTE|PREMESSO|RILEVAT[OA]|CONSIDERAT[OA]|OSSERVA|RITENUT[OA]|VIST[OAIE]\b|ESAMINAT[OA]|"
                     r"TUTTO CI[OÒ] PREMESSO|CI[OÒ] PREMESSO|PER QUESTI MOTIVI|DICHIARA|ORDINA|INGIUNGE|DISPONE|"
                     r"IL PRESIDENTE|IL RELATORE|IL SEGRETARIO GENERALE|Roma,? \d|"
                     r"OGGETTO|[0-9]{1,2}\.\s|[a-z]\)\s)", re.I)
SENT_END = re.compile(r"[.;:!?][\"»”]?$")
CONTINUATION = re.compile(r"^[,.;:)\]»”]")  # a line starting with punctuation continues the previous one
# Sentence boundary: end punctuation, then a capital. A full stop after an abbreviation ("art. 5", "d.lgs. n. 196",
# "S.p.A. Il") is not a boundary; see sentences().
SENT_BOUNDARY = re.compile(r"[.;!?][\"»”)]*\s+(?=[A-ZÀÈÉÌÒÙ«“\"])")
ABBREVIATIONS = set("""art artt par parr lett n nn comma cfr pag pagg prot cass sez cit ss segg ecc dott ssa prof sig
sigg avv on reg rif all cd l p pp co es vol ord sent civ pen proc cost ivi op spa srl lgs dlgs dpr c doc ult succ mod
cap tab nr num rg gu dl cons trib giur dir sg vd v ca""".split())
# Fallback for a single sentence longer than CHUNK_TOKENS: also split before numbers and opening brackets.
CLAUSE_SPLIT = re.compile(r"(?<=[.;!?])\s+(?=[A-ZÀÈÉÌÒÙ«“\"(\[0-9])")
# A sentence saying what was violated is kept even when its wording is formulaic: "per la violazione degli artt. 5
# e 13", "ha violato l'art. 32", "in violazione del principio di minimizzazione" (but not "l'annotazione delle
# violazioni e delle misure adottate", the register formula).
VIOLATION_STATEMENT = re.compile(r"\b(?:violazion[ei]|violat[oaie])\s+(?:di quanto (?:previsto|disposto|stabilito)\s+"
                                 r"(?:da(?:l|ll[’']|gli|i|lla|lle)?\s*)?)?(?:de(?:gli|i|l|lla|lle|ll[’'])\s*|gli\s+|"
                                 r"i\s+|l[’']\s*|la\s+|le\s+|il\s+)?(?:art|principi|disposizion|obblig)", re.I)

# Sections of a decision. Openers are matched case-sensitively (they are upper case in the decisions);
# paragraphs without an opener inherit the current section.
SECTION_NAMES = {"sintesi": "Sintesi", "fatti": "Fatto", "motivazione": "Motivazione", "dispositivo": "Decisione",
                 "normativa": "Quadro normativo", "procedura": "Atti del procedimento", "testo": "Testo"}
# Search weights: the case's activity (sintesi, fatti) and the violation (motivazione, dispositivo) rank first.
SECTION_WEIGHTS = {"sintesi": 1.0, "fatti": 1.0, "motivazione": 1.0,
                   "dispositivo": 0.97, "normativa": 0.97, "testo": 0.97, "procedura": 0.9}
ROLES = {"sintesi": "Attività", "fatti": "Attività", "motivazione": "Violazione", "dispositivo": "Violazione"}
# Section scopes for search and the embedding map: a chunk is in scope when its majority section has that role.
SCOPES = {"violation": "Violazione", "activity": "Attività"}
# Decisions on a case. Only these can find a violation, so the violation scope leaves out opinions on draft laws,
# deliberations, general provisions, authorisations, prior checks, research authorisations (art. 110), guidelines
# and the like. "Decisione su ricorso" are the pre-2018 decisions on complaints (art. 145 of the old Codice).
CASE_TYPES = {"Ordinanza ingiunzione o revoca", "Prescrizioni del Garante", "Ammonimento", "Avvertimento",
              "Prescrizioni e divieto del Garante", "Divieto del trattamento", "Blocco del trattamento", "Provvedimenti",
              "Decisione su ricorso"}
DROP_SECTIONS = {"preambolo", "firma"}  # attendance list ("in presenza del prof. ...") and signatures
OPEN_PREAMBOLO = re.compile(r"^IL GARANTE PER LA PROTEZIONE DEI DATI PERSONALI\b")
OPEN_FIRMA = re.compile(r"^(IL PRESIDENTE|IL RELATORE|IL SEGRETARIO GENERALE|IL VICE ?PRESIDENTE|Roma,? \d)")
OPEN_PREMESSO_GARANTE = re.compile(r"^(TUTTO )?CI[OÒ] PREMESSO,? IL GARANTE")
OPEN_DISPOSITIVO = re.compile(r"^(PER QUESTI MOTIVI|DICHIARA|ORDINA|INGIUNGE|DISPONE|PRESCRIVE|VIETA|ACCOGLIE)\b")
OPEN_FATTI = re.compile(r"^(PREMESSO|RILEVAT[OA]|ESAMINAT[OA])\b")
OPEN_MOTIVAZIONE = re.compile(r"^(CONSIDERAT[OA]|RITENUT[OA]|OSSERVAT[OA]|OSSERVA|TENUTO CONTO)\b")
OPEN_VISTO = re.compile(r"^VIST[OAIE]\b")
# GDPR-era decisions number their sections and sub-sections: "1. Introduzione.", "3. VALUTAZIONI DELL'AUTORITÀ",
# "4.1 Assenza di una base giuridica (artt. 6 e 9)", "5. Adozione dell'ordinanza ingiunzione ... (artt. 58, par. 2,
# lett. i; 83 RGPD)". A heading reads as a title, not a sentence: numbered, capitalised, short once citations in
# brackets are removed, without a finite verb, and not ending with ; : , (those are numbered points and list items).
NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+(?=[A-ZÀ-Ú«“\"'’])")
SENTENCE_VERB = re.compile(r"\b(ha|hanno|è|sono|era|erano|sia|siano|risulta|risultano|ritiene|rileva|dichiara|lamenta|"
                           r"può|possono|deve|devono|viene|vengono|venivano|si|aveva|avevano|avveniva|inviava)\b", re.I)
HEADING_MAX_CHARS, HEADING_MAX_WORDS = 250, 30
# The heading's words decide the section (first match wins; the first section of a decision is its facts).
# "Conclusioni: dichiarazione di illiceità ... Provvedimenti correttivi" and "L'esito dell'istruttoria" open the
# reasoning even when they go on to name the measures; the procedure's own name ("procedimento per l'adozione dei
# provvedimenti correttivi") says nothing about where the decision is.
HEADING_PROCEDURE = re.compile(r"(procedimento\s+)?per\s+l[’']\s*adozione\s+de(i|gli|l)\s+provvedimenti?\s+"
                               r"(correttiv|sanzionator)\w*(\s+e\s+(correttiv|sanzionator)\w*)?", re.I)
HEADING_SECTIONS = [
    ("motivazione", re.compile(r"^\S+\s+(l[’']\s*)?(conclusion|esit)", re.I)),
    ("dispositivo", re.compile(r"ordinanza|ingiunzion|applicazione dell[ae] sanzion|quantificazion|"
                               r"determinazione dell[ae] sanzion|misure correttive|provvedimenti correttivi|"
                               r"poteri correttivi|corrective|ammonimento", re.I)),
    ("motivazione", re.compile(r"esit|valutazion|contestazion|illiceit|illegittimit|conclusion|osservazion|difes|memori|"
                               r"argomentazion|base giuridic|basi giuridic|titolarit|profili|risultanz|"
                               r"ambito di applicazione|findings|inosservanz|mancat[oaie]\s|principi[oi]?\s+d|obblig|"
                               r"violazion\w*\s+(?:de(?:gli|i|l|lla|lle|ll[’'])\s*|di cui\s+)?"
                               r"(?:art|princip|obblig|disposiz)|"
                               r"violazion\w*\s+(?:accertat|contestat|rilevat|riscontrat)|\(artt?\.\s*\d|"
                               r"^\S+\s+sul(l[a'’o]|l?e|\s)", re.I)),
    ("normativa", re.compile(r"quadro (giuridico|normativo)|(normativa|disciplina) (applicabile|di riferimento)", re.I)),
    ("fatti", re.compile(r"introduzion|premess|reclam|istruttori|segnalazion|richiest|ispett|accertament|notific|"
                         r"violazione de?i dati|data breach|misure|ricostruzion|istanz|il caso|il fatto|i fatti|"
                         r"trattamento|svolgimento|procedimento|fact-finding", re.I)),
]
HEADING_FACTS = HEADING_SECTIONS[-1][1]
# Headings move a decision forward through facts -> reasoning -> decision, never back: a sub-heading inside the
# reasoning such as "4.2 Il trattamento effettuato" names the matter discussed, not a return to the facts.
SECTION_STAGE = {"fatti": 0, "normativa": 0, "motivazione": 1, "dispositivo": 2}
DROP_PARAGRAPH = re.compile(r"^Registro dei provvedimenti\b")
# English translations: about a hundred decisions are published only in English, and some recent pages carry the
# Italian decision followed by its English translation. Both are left out: they duplicate Italian text and pull
# English queries towards translations rather than towards the matter asked about. A paragraph is English when it
# has more English than Italian function words; the translation is the tail of the page that is at least
# TRANSLATION_MIN_SHARE English (so a short English quote inside an Italian decision stays).
ENGLISH_WORDS = re.compile(r"\b(the|and|of|to|is|that|with|for|be|which|this|are|by)\b")
ITALIAN_WORDS = re.compile(r"\b(il|la|di|che|e|del|della|per|con|non|dei|delle|un|una)\b")
TRANSLATION_MIN_SHARE, TRANSLATION_MIN_WORDS = 0.8, 150
# A VISTO paragraph citing a law states what the case is about (e.g. art. 144-bis on intimate images), so it is
# legal framework rather than procedure. Generic citations (the GDPR, the Codice) are removed as boilerplate.
VISTO_NORMA = re.compile(r"^VIST[OAIE]\s+(altresì\s+)?(l[’'a]\s*|il\s+|i\s+|gli\s+|le\s+)?"
                         r"((medesim|citat|predett)[oaie]\s+)?(art|legge|d\.?\s?l|decret|"
                         r"codice|regolament|direttiva|d\.p\.r|provvedimento|parere|linee guida)", re.I)
# A VISTO paragraph that reports what a party did or claimed is part of the facts, not procedure.
VISTO_FACTS = re.compile(r"con (la|il|le|i) qual[ei]|lamenta|ha chiesto|chiede|ha rappresentato|ha dichiarato|"
                         r"ha affermato|ha precisato|ha sostenuto|ha segnalato|ha contestato|deduce", re.I)


# ----------------------------------------------------------------------------- cleaning
def heading_number(p: str) -> str | None:
    """The number ("4.1") when the paragraph is a numbered heading, else None."""
    m = NUMBERED.match(p)
    if not m or len(p) > HEADING_MAX_CHARS or p.rstrip()[-1:] in (";", ":", ","):
        return None
    title = re.sub(r"\([^()]*\)", " ", p[m.end():])  # "(artt. 58, par. 2, lett. i; 83 RGPD)"
    if len(title.split()) > HEADING_MAX_WORDS or SENTENCE_VERB.search(title):
        return None
    return m.group(1)


def heading_section(p: str, number: str, current: str) -> str:
    p = HEADING_PROCEDURE.sub(" ", p)
    target = next((s for s, rx in HEADING_SECTIONS if rx.search(p)), None)
    if number.split(".")[0] == "1" and HEADING_FACTS.search(p):
        target = "fatti"  # "1. La segnalazione pervenuta, l'attività istruttoria e i relativi esiti"
    if target is None or SECTION_STAGE[target] < SECTION_STAGE.get(current, 0):
        return current
    return target


def label_sections(paras: list[str], title: str) -> list[tuple[str, str]]:
    """(section, paragraph) pairs; text before the preamble is the editorial summary when the doc has one."""
    has_preamble = any(OPEN_PREAMBOLO.match(p) for p in paras)
    bare_title = re.sub(r"\s*\[\d+\]\s*$", "", title).strip()
    section = "sintesi" if has_preamble else "testo"
    out = []
    for p in paras:
        if DROP_PARAGRAPH.match(p):
            continue
        number = heading_number(p)
        if number and section not in ("preambolo", "firma"):
            section = heading_section(p, number, section)
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


def strip_translation(paras: list[str]) -> list[str]:
    """The paragraphs before an English translation at the end of the page (all of them when there is none)."""
    cut, words, english = len(paras), 0, 0
    for i in range(len(paras) - 1, -1, -1):
        n = len(paras[i].split())
        is_en = len(ENGLISH_WORDS.findall(paras[i].lower())) > len(ITALIAN_WORDS.findall(paras[i].lower()))
        words += n
        english += n if is_en else 0
        if is_en and english >= TRANSLATION_MIN_WORDS and english >= TRANSLATION_MIN_SHARE * words:
            cut = i
    return paras[:cut]


def paragraphs(body: str) -> list[str]:
    """Rejoin lines broken at inline HTML elements into real paragraphs; an English translation is left out."""
    body = DOCWEB_HEADER.sub("", body.strip()).replace("´", "'").replace("`", "'")
    paras, cur = [], []
    for line in body.splitlines():
        line = " ".join(line.split())
        if not line:
            continue
        if cur and not CONTINUATION.match(line) and (HEADING.match(line) or SENT_END.search(cur[-1])
                                                     or heading_number(" ".join(cur))):
            paras.append(" ".join(cur))
            cur = []
        cur.append(line)
    if cur:
        paras.append(" ".join(cur))
    # tidy spacing left by the line joins: "( testo )", "parola ,"
    return strip_translation([re.sub(r"\s+([,.;:)\]])", r"\1", re.sub(r"([(\[])\s+", r"\1", p)) for p in paras])


def sentences(p: str) -> list[str]:
    out, start = [], 0
    for m in SENT_BOUNDARY.finditer(p):
        if p[m.start()] == ".":
            prev = re.search(r"[\w’']+$", p[start:m.start()])
            w = prev.group(0).lower() if prev else ""
            if w in ABBREVIATIONS or len(w) == 1 or w.isdigit() or (prev and prev.group(0).isupper() and len(w) <= 3):
                continue  # "art.", "n.", "S.p.A.", "UE.", "12. Conclusioni"
        out.append(p[start:m.end()].strip())
        start = m.end()
    out.append(p[start:].strip())
    return [s for s in out if s]


def select_docs(n: int = 0) -> tuple[list[dict], Counter]:
    """The newest n usable documents (all of them when n is 0), newest first."""
    recs = [json.loads(line) for line in (DATA / "docs.jsonl").open(encoding="utf-8")]
    recs.sort(key=lambda r: (r["date"] or "", r["docweb_id"]), reverse=True)
    dropped, seen, out = Counter(), set(), []
    for r in recs:
        if n and len(out) == n:
            break
        if r["body_chars"] < MIN_BODY_CHARS:
            dropped["short_or_empty"] += 1
        elif STUB.search(r["body"]):
            dropped["redirect_stub"] += 1
        elif r["body_sha256"] in seen:
            dropped["duplicate_body"] += 1
        elif len(" ".join(paragraphs(r["body"]))) < MIN_BODY_CHARS:
            dropped["english"] += 1  # an English translation, nothing (or only a title) in Italian
        else:
            seen.add(r["body_sha256"])
            out.append(r)
    return out, dropped


# ----------------------------------------------------------------------------- templates + boilerplate
def _shingles(body: str):
    """Sorted distinct 5-word shingles as 64-bit hashes. The hash is stable across runs (Python's hash() of a string
    is salted per process), so template families come out the same every time."""
    import numpy as np
    text = re.sub(r"\d+", "0", body.lower())
    text = re.sub(r"\bx{2,}\b|\[omissis\]", "x", text)  # anonymised names
    words = re.findall(r"\w+", text)
    digest = lambda s: int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
    return np.unique(np.fromiter((digest(" ".join(words[i:i + 5])) for i in range(max(1, len(words) - 4))),
                                 dtype=np.uint64))


def near_duplicate_clusters(docs: list[dict]) -> list[list[int]]:
    """Groups of near-identical decisions (template families such as the "revenge porn" ratifications), found
    with MinHash + LSH and confirmed by exact Jaccard >= DUP_JACCARD. Indices into `docs`, each group in input
    order (so with docs sorted newest first, the first index is the newest member and becomes the representative)."""
    import numpy as np
    sets = [_shingles(d["body"]) for d in docs]  # arrays, not sets: a tenth of the memory on the full corpus
    rng = np.random.default_rng(0)
    prime = np.uint64((1 << 61) - 1)
    a = rng.integers(1, int(prime), 128, dtype=np.uint64)
    b = rng.integers(0, int(prime), 128, dtype=np.uint64)
    sig = np.zeros((len(sets), 128), dtype=np.uint64)
    for i, sh in enumerate(sets):
        h = sh & np.uint64(0xFFFFFFFFFFFF)  # 48 bits
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
                inter = len(np.intersect1d(sets[pair[0]], sets[pair[1]], assume_unique=True))
                if inter / (len(sets[pair[0]]) + len(sets[pair[1]]) - inter) >= DUP_JACCARD:
                    parent[find(pair[0])] = find(pair[1])
    groups: dict[int, list[int]] = {}
    for i in range(len(sets)):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _is_boilerplate(p: str, keys: frozenset[str] | set[str]) -> bool:
    # numbered headings repeat across decisions but give each chunk its context, so they always stay
    return len(p) >= BOILERPLATE_MIN_CHARS and not heading_number(p) and _para_key(p) in keys


def _para_key(p: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "0", p.lower())).strip()


def boilerplate_keys(docs: list[dict]) -> set[str]:
    """Paragraphs repeated in at least BOILERPLATE_MIN_DOCS decisions ("VISTO il Regolamento (UE) 2016/679 ...",
    "VISTA la documentazione in atti", the notice on how to appeal). Count over template representatives only,
    so a family of 700 copies does not turn its own subject matter into boilerplate."""
    df = Counter()
    for doc in docs:
        df.update({_para_key(p) for _, p in label_sections(paragraphs(doc["body"]), doc["title"])
                   if len(p) >= BOILERPLATE_MIN_CHARS and not heading_number(p)})
    return {k for k, n in df.items() if n >= BOILERPLATE_MIN_DOCS}


def _phrases(text: str) -> set[int]:
    """5-word phrases with numbers and anonymised names normalised (hashes; salted per process, so not stored)."""
    words = re.findall(r"\w+", re.sub(r"\bx{2,}\b|\[omissis\]|\.{3,}", "x", re.sub(r"\d+", "0", text.lower())))
    return {hash(" ".join(words[i:i + 5])) & 0x7FFFFFFFFFFFFFFF for i in range(max(1, len(words) - 4))}


def procedural_phrases(docs: list[dict], boilerplate: frozenset[str]) -> frozenset[int]:
    """Phrases in at least BOILERPLATE_MIN_DOCS decisions (template representatives) that no single topic tag
    (`argomenti`) dominates among the tagged decisions using them: procedural wording rather than the vocabulary
    of one kind of case."""
    import numpy as np
    from scipy import sparse
    hashes, owners = [], []  # each document's distinct phrases, as arrays (a set of pairs does not fit in memory)
    for i, doc in enumerate(docs):
        ph = set()
        for _, p in label_sections(paragraphs(doc["body"]), doc["title"]):
            if not _is_boilerplate(p, boilerplate):
                for s in sentences(p):
                    ph |= _phrases(s)
        hashes.append(np.fromiter(ph, dtype=np.int64, count=len(ph)))
        owners.append(np.full(len(ph), i, dtype=np.int32))
    hashes, owners = np.concatenate(hashes), np.concatenate(owners)
    distinct, df = np.unique(hashes, return_counts=True)
    frequent = distinct[df >= BOILERPLATE_MIN_DOCS]
    keep = np.isin(hashes, frequent)
    in_doc = sparse.csr_matrix((np.ones(int(keep.sum()), np.float32),
                                (np.searchsorted(frequent, hashes[keep]), owners[keep])), shape=(len(frequent), len(docs)))
    tags = {t: j for j, t in enumerate(sorted({t for d in docs for t in d["argomenti"]}))}
    rows, cols = zip(*((i, tags[t]) for i, d in enumerate(docs) for t in d["argomenti"]))
    tagged = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(docs), len(tags)))
    n_tagged = in_doc @ np.array([bool(d["argomenti"]) for d in docs], dtype=np.float32)
    top_tag = np.asarray((in_doc @ tagged).max(axis=1).todense()).ravel()
    spread = (n_tagged >= FORMULA_MIN_TAGGED) & (top_tag <= FORMULA_MAX_TOPIC * n_tagged)
    return frozenset(frequent[spread].tolist())


def is_formula(sentence: str, procedural: frozenset[int]) -> bool:
    if not procedural or VIOLATION_STATEMENT.search(sentence):
        return False
    if heading_number(sentence) and re.search(r"\bartt?\.", sentence, re.I):
        return False  # a heading naming an article ("3.6.1 ART. 5, PAR. 1 LETT. A) ...") says what is discussed
    ph = _phrases(sentence)
    return len(ph & procedural) >= FORMULA_MIN_SHARE * len(ph)


# ----------------------------------------------------------------------------- chunking
def header(doc: dict, section: str, heading: str = "") -> str:
    parts = [doc["title"], f"Data: {doc['date']}"]
    if doc["tipologia"]:
        parts.append("Tipologia: " + ", ".join(doc["tipologia"]))
    if doc["argomenti"]:
        parts.append("Argomenti: " + ", ".join(doc["argomenti"]))
    parts.append("Sezione: " + SECTION_NAMES[section])
    if heading:
        parts.append("Paragrafo: " + heading)
    return " | ".join(parts)


def chunk_doc(doc: dict, ntok, boilerplate: frozenset[str] = frozenset(), procedural: frozenset[int] = frozenset(),
              dropped: Counter | None = None) -> list[tuple[str, dict[str, float], str]]:
    """(text, section token shares, heading path) per chunk. Chunks follow the decision's own structure: a new
    section or numbered sub-section ("4.1 Assenza di una base giuridica") starts a new chunk once the current one
    has MIN_SECTION_TOKENS, so a small sub-section joins the next one and a long one is split, with overlap only
    inside it. The heading path ("Valutazioni > Assenza di una base giuridica (artt. 6 e 9)") goes with every chunk
    of the sub-section; formulaic headings ("Conclusioni", "Adozione dell'ordinanza ingiunzione ...") are left out
    of it. Procedural formula sentences are left out of the text; `dropped` counts their words by section."""
    # Units are sentences grouped by paragraph; a paragraph longer than the budget is split into sentences.
    units: list[tuple[str, int, bool, str, str]] = []  # (text, tokens, starts_paragraph, section, sub-section)
    stack: list[tuple[int, str, str]] = []  # open headings: (depth, number, title or "" when formulaic)
    for section, p in label_sections(paragraphs(doc["body"]), doc["title"]):
        number = heading_number(p)
        if number:
            depth = number.count(".") + 1
            stack = [h for h in stack if h[0] < depth]
            title = NUMBERED.sub("", p).rstrip(" .")
            stack.append((depth, number, "" if is_formula(p, procedural) else title))
        if _is_boilerplate(p, boilerplate):
            continue
        sents = sentences(p)
        kept = [s for s in sents if not is_formula(s, procedural)]
        if dropped is not None:
            dropped[section] += sum(len(s.split()) for s in sents if s not in kept)
        if not kept:
            continue
        p = " ".join(kept)
        if ntok(p) <= CHUNK_TOKENS:
            sents = [p]
        else:
            sents = [c for s in kept for c in ([s] if ntok(s) <= CHUNK_TOKENS else CLAUSE_SPLIT.split(s))]
        part = " > ".join(t for _, _, t in stack if t) + "\x00" + (stack[-1][1] if stack else "")
        for i, s in enumerate(sents):
            units.append((s, ntok(s), i == 0, section, part))
    size = lambda c: sum(u[1] for u in c)
    chunks: list[list[tuple[str, int, bool, str, str]]] = []
    cur: list[tuple[str, int, bool, str, str]] = []
    for u in units:
        new_part = bool(cur) and u[3:] != cur[-1][3:] and size(cur) >= MIN_SECTION_TOKENS
        if cur and (new_part or size(cur) + u[1] > CHUNK_TOKENS):
            chunks.append(cur)
            tail, n = [], 0  # overlap: trailing units of the previous chunk, up to OVERLAP_TOKENS
            for prev in reversed(cur):
                if new_part or prev[3:] != u[3:] or n + prev[1] > OVERLAP_TOKENS:
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
        text, shares, parts = "", Counter(), Counter()
        for s, t, starts_para, section, part in c:
            text += ("\n" if starts_para and text else " " if text else "") + s
            shares[section] += t
            parts[part] += t
        total = sum(shares.values())
        heading = parts.most_common(1)[0][0].split("\x00")[0]  # the sub-section with most of the chunk
        out.append((text, {k: round(v / total, 3) for k, v in shares.most_common()}, heading))
    return out


def cmd_chunk(args) -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ntok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    docs, dropped = select_docs(args.n)
    if args.n and len(docs) < args.n:
        print(f"[warn] only {len(docs)} usable documents")
    OUT.mkdir(parents=True, exist_ok=True)
    # Template families: index the newest member, list the others with it (shown as "similar decisions").
    clusters = near_duplicate_clusters(docs)
    reps = [docs[g[0]] for g in clusters]
    similar = {docs[g[0]]["docweb_id"]: [{k: docs[i][k] for k in ("docweb_id", "date", "title", "url")}
                                         for i in g[1:]] for g in clusters if len(g) > 1}
    (OUT / "duplicates.json").write_text(json.dumps(similar, ensure_ascii=False), encoding="utf-8")
    boilerplate = frozenset(boilerplate_keys(reps))
    procedural = procedural_phrases(reps, boilerplate)
    n_chunks, sizes, sections, formula_words = 0, [], Counter(), Counter()
    with (OUT / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for doc in reps:
            n_similar = len(similar.get(doc["docweb_id"], []))
            for i, (text, shares, heading) in enumerate(chunk_doc(doc, ntok, boilerplate, procedural, formula_words)):
                section = next(iter(shares))  # majority section by tokens
                rec = {"chunk_id": f"{doc['docweb_id']}_{i:03d}", "docweb_id": doc["docweb_id"], "chunk_index": i,
                       "title": doc["title"], "date": doc["date"], "url": doc["url"],
                       "tipologia": doc["tipologia"], "argomenti": doc["argomenti"],
                       "section": section, "section_shares": shares, "heading": heading, "similar_count": n_similar,
                       "weight": round(sum(SECTION_WEIGHTS[k] * v for k, v in shares.items()), 4),
                       "text": text, "embed_text": f"{header(doc, section, heading)}\n\n{text}"}
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
            "procedural_phrases": len(procedural),
            "formula_words_dropped_by_section": dict(formula_words.most_common()),
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
    model_kwargs = dict(spec.get("model_kwargs", {}))
    if device == "cuda":
        model_kwargs["dtype"] = getattr(torch, spec.get("dtype", "float16"))
    if "revision" not in spec:
        model = SentenceTransformer(spec["name"], device=device, model_kwargs=model_kwargs)
    else:
        model_kwargs["trust_remote_code"] = True
        model = SentenceTransformer(spec["name"], revision=spec["revision"], device=device, trust_remote_code=True,
                                    model_kwargs=model_kwargs, tokenizer_kwargs=spec.get("tokenizer_kwargs"))
    model.max_seq_length = 1024
    return model


def encode_queries(model, key: str, texts: list[str]):
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, **MODELS[key]["query"])


def encode_docs(model, key: str, texts: list[str], batch_size: int | None = None):
    batch_size = batch_size or MODELS[key].get("batch_size", 16)
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                        **MODELS[key]["doc"])


def embeddings_path(key: str = DEFAULT_MODEL) -> Path:
    return OUT / MODELS[key]["file"]


# ----------------------------------------------------------------------------- BM25 + hybrid
BM25_K1, BM25_B = 1.2, 0.75
RRF_K = 60              # reciprocal rank fusion constant
HYBRID_ALPHA = 0.3      # share of BM25 in the fused ranking (0 = embeddings only, 1 = BM25 only)
# Queries with exact references (article/decision numbers, quoted phrases) are what keyword search is for:
# at the default share the embedding ranking would bury an exact "art. 144-bis" match.
EXACT_QUERY = re.compile(r"\d|[\"“”«»]")
EXACT_QUERY_MIN_ALPHA = 0.6
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


def effective_alpha(query: str, alpha: float) -> float:
    """BM25 share actually used: raised to EXACT_QUERY_MIN_ALPHA for queries with numbers or quotes."""
    return max(alpha, EXACT_QUERY_MIN_ALPHA) if 0 < alpha and EXACT_QUERY.search(query) else alpha


def score_chunks(q_vec, query: str, emb, bm25: BM25 | None, weights=None, alpha: float = HYBRID_ALPHA):
    """Chunk scores for a query: section-weighted cosine, fused with section-weighted BM25 when alpha > 0
    (with a larger BM25 share for exact-reference queries, see effective_alpha)."""
    dense = emb @ q_vec
    if weights is not None:
        dense = dense * weights
    if bm25 is None or alpha <= 0:
        return dense
    lexical = bm25.scores(query)
    if weights is not None:
        lexical = lexical * weights
    return fuse(dense, lexical, effective_alpha(query, alpha))


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


def scope_mask(chunks: list[dict], scope: str | None):
    """Bool array over chunks: those whose majority section belongs to `scope` (a SCOPES key); None = all chunks.
    The violation scope also requires a case decision (CASE_TYPES)."""
    import numpy as np
    if scope is None:
        return np.ones(len(chunks), dtype=bool)
    role = SCOPES[scope]
    return np.array([ROLES.get(c.get("section", "testo")) == role
                     and (scope != "violation" or not CASE_TYPES.isdisjoint(c["tipologia"])) for c in chunks])


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
    mask = scope_mask(chunks, args.sections) if args.sections else None
    for rank, (d, hits) in enumerate(rank_documents(scores, chunks, args.k, mask=mask).items(), 1):
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
    p.add_argument("--n", type=int, default=0, help="number of newest documents to include (default 0: all)")
    p = sub.add_parser("embed")
    p.add_argument("--batch-size", type=int, default=None, help="default: the model's own, else 16")
    p.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--no-weights", action="store_true", help="ignore section weights (plain cosine)")
    p.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    p.add_argument("--alpha", type=float, default=HYBRID_ALPHA, help="BM25 share in the fusion (0 = embeddings only)")
    p.add_argument("--sections", choices=SCOPES, default=None,
                   help="search only these sections: violation (motivazione, dispositivo of case decisions) "
                        "or activity (sintesi, fatti)")
    sub.add_parser("bm25")
    args = ap.parse_args()
    {"chunk": cmd_chunk, "embed": cmd_embed, "bm25": cmd_bm25, "search": cmd_search}[args.cmd](args)


if __name__ == "__main__":
    main()
