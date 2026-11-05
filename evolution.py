#!/usr/bin/env python3
"""
evolution.py - the versions of the search index, side by side: what the "Evoluzione del progetto" page of the app
(pages/1_Evoluzione_del_progetto.py) runs its before/after comparisons on.

Each version is a snapshot of data/poc kept in its own folder (data/poc/baseline, data/poc/v1_2004_2018, ...;
the current index is data/poc itself), with the chunks, the embeddings of the models tried on it and, from v3 on,
the BM25 index. A snapshot is opened lean, so that several fit in memory next to the search page: per passage only
what ranking needs (document, section, weight) and where its line starts in chunks.jsonl, whose text is read back
only for the passages shown; the embeddings stay on disk (memory-mapped float16).

  python evolution.py eval      # retrieval scores of every stage -> data/poc/evolution_eval.json (shown by the page)
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import build_index as b

EVAL_PATH = b.OUT / "evolution_eval.json"
BLOCK = 16384  # embedding rows converted to float32 at a time when scoring


# ----------------------------------------------------------------------------- snapshots
@dataclass
class Snapshot:
    name: str                   # folder under data/poc; "" is the current index
    path: Path
    meta: dict
    chunks: list[dict]          # lean: docweb_id, section, tipologia, heading, offset of the line in chunks.jsonl
    docs: dict[int, dict]       # docweb_id -> title, date, url, tipologia, argomenti
    weights: np.ndarray         # section weights per chunk (1.0 before sections were labelled)
    similar: dict[int, list]    # template representative -> its near-identical decisions
    _emb: dict[str, np.ndarray] = field(default_factory=dict)

    def models(self) -> list[str]:
        return [m for m in b.MODELS if (self.path / b.MODELS[m]["file"]).exists()]

    def has_bm25(self) -> bool:
        return (self.path / "bm25.npz").exists()

    def embeddings(self, model: str) -> np.ndarray:
        if model not in self._emb:
            emb = np.load(self.path / b.MODELS[model]["file"], mmap_mode="r")
            if len(emb) != len(self.chunks):
                raise RuntimeError(f"{self.path}: {len(emb)} {model} vectors for {len(self.chunks)} passages")
            self._emb[model] = emb
        return self._emb[model]

    def text(self, i: int) -> str:
        with (self.path / "chunks.jsonl").open("rb") as fh:
            fh.seek(self.chunks[i]["offset"])
            return json.loads(fh.readline())["text"]

    def doc(self, i: int) -> dict:
        return self.docs[self.chunks[i]["docweb_id"]]


def open_snapshot(name: str) -> Snapshot:
    path = b.OUT / name if name else b.OUT
    chunks, docs, weights, offset = [], {}, [], 0
    with (path / "chunks.jsonl").open("rb") as fh:
        for line in fh:
            c = json.loads(line)
            d = docs.get(c["docweb_id"])
            if d is None:
                d = docs[c["docweb_id"]] = {k: c[k] for k in ("docweb_id", "title", "date", "url", "tipologia",
                                                              "argomenti")}
            chunks.append({"docweb_id": c["docweb_id"], "section": c.get("section", "testo"), "tipologia": d["tipologia"],
                           "heading": c.get("heading", ""), "offset": offset})
            weights.append(c.get("weight", 1.0))
            offset += len(line)
    dup = path / "duplicates.json"
    similar = {int(k): v for k, v in json.loads(dup.read_text(encoding="utf-8")).items()} if dup.exists() else {}
    meta = json.loads((path / "chunks_meta.json").read_text(encoding="utf-8"))
    return Snapshot(name, path, meta, chunks, docs, np.array(weights, dtype=np.float32), similar)


def open_bm25(snapshot: Snapshot) -> b.BM25 | None:
    return b.BM25(snapshot.path) if snapshot.has_bm25() else None


# ----------------------------------------------------------------------------- configurations
@dataclass(frozen=True)
class Config:
    """One way of searching one version of the index."""
    snapshot: str               # folder under data/poc, "" = current index
    model: str                  # key of build_index.MODELS
    alpha: float = 0.0          # BM25 share in the fused ranking (0 = embeddings only)
    weights: bool = False       # section weights
    scope: str | None = None    # None | "violation" | "activity"
    label: str = ""

    def describe(self) -> str:
        parts = [b.MODELS[self.model]["label"]]
        if self.alpha > 0:
            parts.append("solo BM25" if self.alpha >= 1 else f"BM25 {self.alpha:.0%}")
        if self.weights:
            parts.append("pesi per sezione")
        if self.scope == "violation":
            parts.append("solo violazioni")
        elif self.scope == "activity":
            parts.append("solo fatti")
        return " · ".join(parts)


def cosine(emb: np.ndarray, q: np.ndarray) -> np.ndarray:
    """emb @ q for float16 (memory-mapped) embeddings, a block at a time in float32."""
    q = q.astype(np.float32)
    out = np.empty(len(emb), dtype=np.float32)
    for s in range(0, len(emb), BLOCK):
        out[s:s + BLOCK] = np.asarray(emb[s:s + BLOCK], dtype=np.float32) @ q
    return out


def score(cfg: Config, snap: Snapshot, bm25: b.BM25 | None, q_vec: np.ndarray, query: str) -> np.ndarray:
    """Chunk scores, as build_index.score_chunks computes them."""
    dense = cosine(snap.embeddings(cfg.model), q_vec)
    w = snap.weights if cfg.weights else None
    if w is not None:
        dense = dense * w
    if cfg.alpha <= 0 or bm25 is None:
        return dense
    lexical = bm25.scores(query)
    if w is not None:
        lexical = lexical * w
    return b.fuse(dense, lexical, b.effective_alpha(query, cfg.alpha))


def search(cfg: Config, snap: Snapshot, bm25: b.BM25 | None, q_vec: np.ndarray, query: str, k: int = 10):
    """{docweb_id: {"top"|"Attività"|"Violazione": (score, chunk index)}}, best first."""
    mask = b.scope_mask(snap.chunks, cfg.scope) if cfg.scope else None
    return b.rank_documents(score(cfg, snap, bm25, q_vec, query), snap.chunks, k, mask=mask)


# ----------------------------------------------------------------------------- evaluation
def config_key(cfg: Config) -> str:
    return f"{cfg.snapshot or 'current'}|{cfg.model}|{cfg.alpha:g}|{int(cfg.weights)}|{cfg.scope or 'all'}"


def evaluate(cfg: Config, snap: Snapshot, bm25: b.BM25 | None, q_vecs: np.ndarray) -> dict:
    """P@10 and reciprocal rank per query of eval_retrieval.QUERIES (the Garante's topic tags as weak labels),
    and their means over the Italian queries, the English ones and all of them."""
    import eval_retrieval as ev
    rows = []
    for (query, tags), q in zip(ev.QUERIES, q_vecs):
        ranked = search(cfg, snap, bm25, q, query, ev.K)
        rel = [bool(tags & set(snap.doc(h["top"][1])["argomenti"])) for h in ranked.values()]
        rows.append((sum(rel) / ev.K, next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0)))
    out = {"per_query": [[round(p, 3), round(rr, 3)] for p, rr in rows]}
    for name, sl in (("it", slice(0, 15)), ("en", slice(15, None)), ("all", slice(None))):
        sub = rows[sl]
        out[name] = [round(float(np.mean([r[0] for r in sub])), 4), round(float(np.mean([r[1] for r in sub])), 4)]
    return out


def cmd_eval(configs: list[Config]) -> None:
    import eval_retrieval as ev
    results = json.loads(EVAL_PATH.read_text(encoding="utf-8")) if EVAL_PATH.exists() else {}
    snaps: dict[str, tuple[Snapshot, b.BM25 | None]] = {}
    for model in dict.fromkeys(c.model for c in configs):  # one model in memory at a time
        t0 = time.perf_counter()
        m = b.load_model(model)
        q_vecs = b.encode_queries(m, model, [q for q, _ in ev.QUERIES])
        for cfg in (c for c in configs if c.model == model):
            if cfg.snapshot not in snaps:
                snap = open_snapshot(cfg.snapshot)
                snaps[cfg.snapshot] = (snap, open_bm25(snap))
            snap, bm25 = snaps[cfg.snapshot]
            res = results[config_key(cfg)] = evaluate(cfg, snap, bm25, q_vecs)
            print(f"{config_key(cfg):<55} P@10 / MRR  it {res['it'][0]:.3f}/{res['it'][1]:.3f}  "
                  f"en {res['en'][0]:.3f}/{res['en'][1]:.3f}  all {res['all'][0]:.3f}/{res['all'][1]:.3f}", flush=True)
        del m
        print(f"[{model}] {time.perf_counter() - t0:.0f}s", flush=True)
    EVAL_PATH.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"-> {EVAL_PATH}")


# ----------------------------------------------------------------------------- the stages
V0, V1, V2, V3, V4, V5, V6, NOW = ("baseline", "v1_2004_2018", "v2_no_dedup", "v3_verbatim_boilerplate",
                                   "v4_formula_stripped", "v5_subsections_3000", "v6_all_with_english", "")
SNAPSHOT_LABELS = {V0: "v0 · primo indice", V1: "v1 · sezioni", V2: "v2 · provvedimenti 2018–2025",
                   V3: "v3 · senza duplicati né boilerplate", V4: "v4 · senza formule", V5: "v5 · sotto-sezioni",
                   V6: "v6 · tutto l'archivio", NOW: "attuale"}
PARTS = {"ricerca": "Come cerchiamo", "dati": "Cosa indicizziamo", "produzione": "In produzione"}


def _sections(m: dict) -> dict:
    return m.get("chunks_by_section", {})


# Figures read from a snapshot's chunks_meta.json: label and how to get it.
METRICS = {
    "documents": ("Provvedimenti", lambda m: m["documents"]),
    "indexed": ("Provvedimenti indicizzati", lambda m: m.get("indexed_documents", m["documents"])),
    "period": ("Periodo", lambda m: f"{m['date_range'][0][:4]}–{m['date_range'][1][:4]}"),
    "chunks": ("Passaggi", lambda m: m["chunks"]),
    "chunks_per_doc": ("Passaggi per provvedimento", lambda m: m["chunks_per_doc"]),
    "labelled": ("Passaggi con una sezione", lambda m: sum(v for k, v in _sections(m).items() if k != "testo")),
    "families": ("Famiglie di quasi-duplicati", lambda m: m.get("template_families", 0)),
    "boilerplate": ("Paragrafi boilerplate tolti", lambda m: m.get("boilerplate_paragraphs", 0)),
    "procedural": ("Sequenze procedurali", lambda m: m.get("procedural_phrases", 0)),
    "formula_words": ("Parole di formule tolte", lambda m: sum(m.get("formula_words_dropped_by_section", {}).values())),
    "motivazione": ("Passaggi di motivazione", lambda m: _sections(m).get("motivazione", 0)),
    "fatti": ("Passaggi di fatti", lambda m: _sections(m).get("fatti", 0)),
    "testo": ("Passaggi senza sezione", lambda m: _sections(m).get("testo", 0)),
    # v4 stored the formula counts under "dropped" by mistake; it has no English figure either way
    "english": ("Pagine in inglese escluse", lambda m: m.get("dropped", {}).get("english", 0)),
}


def read_meta(snapshot: str) -> dict:
    return json.loads(((b.OUT / snapshot) if snapshot else b.OUT).joinpath("chunks_meta.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Stage:
    key: str
    part: str                           # key of PARTS
    short: str                          # label in the list of stages
    title: str
    summary: str
    points: tuple[str, ...]             # what was done, in Markdown
    before: Config | None = None
    after: Config | None = None
    example: str = ""                   # the query the comparison opens on, the one look_for is about
    look_for: str = ""                  # what to notice in the comparison
    note: str = ""                      # caveat on the comparison
    metrics: tuple[str, ...] = ()       # keys of METRICS, compared between the two snapshots
    example_doc: int | None = None      # one decision shown as it is in both snapshots
    alternatives: tuple[str, ...] = ()  # models the "after" side can switch to


STAGES = [
    Stage(
        "bge", "ricerca", "Solo semantica: BGE-M3", "Il punto di partenza: ricerca semantica con BGE-M3",
        "Il primo prototipo: i provvedimenti divisi in passaggi, ogni passaggio trasformato in un vettore dal modello "
        "multilingue BGE-M3, e la ricerca per somiglianza tra la domanda e i passaggi.",
        ("Primo campione: i provvedimenti dal 2004 al 2018, divisi in passaggi di circa 400 token con 60 token di "
         "sovrapposizione.",
         "Prima dell'embedding, ogni passaggio riceve un'intestazione con titolo, data, tipologia e argomenti del "
         "provvedimento.",
         "La domanda diventa un vettore con lo stesso modello; i provvedimenti sono ordinati per il loro passaggio più "
         "simile (similarità coseno).",
         "BGE-M3 è multilingue: anche una domanda in inglese trova i provvedimenti in italiano."),
        after=Config(V0, "bge-m3"),
        example="telecamere per controllare i dipendenti",
        look_for="La ricerca coglie il significato, non solo le parole: «telecamere per controllare i dipendenti» "
                 "trova la videosorveglianza dei lavoratori, anche chiesta in inglese. Ma i passaggi sono ancora testo "
                 "grezzo, senza struttura: «PRESO ATTO…», «VISTE le memorie…».",
        metrics=("documents", "period", "chunks", "chunks_per_doc")),
    Stage(
        "bm25", "ricerca", "Parole chiave: BM25", "Ricerca ibrida: semantica e parole chiave (BM25)",
        "La ricerca semantica non vede i riferimenti esatti: un articolo, un nome, una sigla. Un indice per parole "
        "chiave (BM25) la affianca e le due classifiche vengono fuse.",
        ("Indice BM25 sui passaggi: minuscole, senza parole vuote, con lo stemming italiano (Snowball).",
         "Le due classifiche si fondono con la *reciprocal rank fusion*: 70% semantica, 30% parole chiave.",
         "Se la domanda contiene numeri o virgolette («art. 144-bis») il peso delle parole chiave sale ad almeno 0,6.",
         "Da sola, la ricerca per parole chiave crolla sulle domande in inglese (grafico a fianco): è un complemento, "
         "non un sostituto."),
        before=Config(V3, "bge-m3", 0.0, True), after=Config(V3, "bge-m3", 0.3, True),
        example="art. 144-bis",
        look_for="L'art. 144-bis del Codice privacy è la norma sul revenge porn. La sola semantica restituisce "
                 "provvedimenti che citano articoli qualsiasi; con BM25 arrivano le deliberazioni sul revenge porn.",
        note="Questa tappa e la successiva usano l'indice v3 (provvedimenti 2018–2025), quello su cui sono stati fatti "
             "i confronti: cambia solo il modo di cercare."),
    Stage(
        "modelli", "ricerca", "Altri modelli di embedding", "Quattro modelli di embedding a confronto",
        "Sugli stessi passaggi abbiamo provato altri tre modelli di embedding e misurato quale porta più provvedimenti "
        "pertinenti nei primi dieci. BidirLM è diventato il modello predefinito.",
        ("**BGE-M3** (568M parametri, 1024 dimensioni) · **Jasper 600M** (distillato da Qwen3-Embedding, 2048) · "
         "**Jina v5 nano** (EuroBERT-210M, 768) · **BidirLM 0.6B** (Qwen3 reso bidirezionale, 1024).",
         "Valutazione su 22 domande (15 in italiano, 7 in inglese): un provvedimento è pertinente se il Garante gli ha "
         "assegnato uno degli argomenti attesi. Misure: quota di pertinenti nei primi 10 (P@10) e posizione del primo "
         "pertinente (MRR).",
         "Gli argomenti sono etichette deboli (editoriali, non sempre complete): i numeri confrontano i modelli tra loro, "
         "non misurano la qualità assoluta.",
         "Jina v5 nano è vicino a BidirLM, ma la sua licenza (CC-BY-NC) esclude l'uso commerciale.",
         "I modelli con codice proprio sono stati verificati e bloccati alla revisione controllata."),
        before=Config(V3, "bge-m3", 0.3, True), after=Config(V3, "bidirlm", 0.3, True),
        alternatives=("bidirlm", "jasper", "jina-v5-nano"),
        example="dati sulle vaccinazioni e green pass durante la pandemia",
        look_for="«Dati sulle vaccinazioni e green pass durante la pandemia» è una delle domande della valutazione: ✓ "
                 "segna i provvedimenti con l'argomento atteso e in testa a ogni colonna c'è la P@10 della domanda.",
        note="Indice v3 (provvedimenti 2018–2025), ricerca ibrida per entrambi i lati."),
    Stage(
        "violazioni", "ricerca", "Solo nelle violazioni", "Cercare solo nelle violazioni",
        "Chi cerca un precedente vuole sapere che cosa è stato contestato. Ora la ricerca può limitarsi ai passaggi in "
        "cui il Garante motiva e decide, nei provvedimenti su casi concreti.",
        ("Ogni passaggio sa a quale sezione appartiene: sintesi, fatti, motivazione, decisione, quadro normativo, atti "
         "del procedimento (tappe 5 e 8).",
         "«Solo violazioni» cerca nella motivazione e nella decisione dei provvedimenti su un caso (ordinanze "
         "ingiunzione, ammonimenti, avvertimenti, prescrizioni, divieti, decisioni su ricorso) e lascia fuori pareri, "
         "deliberazioni, atti generali e verifiche preliminari.",
         "Nella ricerca ogni risultato mostra due passaggi: l'attività (i fatti) e la violazione contestata.",
         "Pesi per sezione: fatti e motivazione 1,0; decisione e quadro normativo 0,97; atti del procedimento 0,9."),
        before=Config(NOW, "bidirlm", 0.3, True), after=Config(NOW, "bidirlm", 0.3, True, "violation"),
        example="conservazione eccessiva delle immagini",
        look_for="«Conservazione eccessiva delle immagini»: sull'intero provvedimento escono soprattutto verifiche "
                 "preliminari (aziende che chiedono di conservare le immagini più a lungo); con «solo violazioni», "
                 "ammonimenti e ordinanze ingiunzione per immagini conservate troppo a lungo.",
        note="Indice attuale, BidirLM con BM25: cambia solo dove si cerca."),
    Stage(
        "sezioni", "dati", "Le sezioni del provvedimento", "Riconoscere le sezioni del provvedimento",
        "Un provvedimento ha una struttura: i fatti, la motivazione, la decisione. Le formule che aprono i paragrafi "
        "permettono di riconoscerla, e ogni passaggio porta con sé la propria sezione.",
        ("PREMESSO, RILEVATO → fatti; CONSIDERATO, RITENUTO → motivazione; DISPONE, ORDINA, INGIUNGE → decisione; "
         "VISTO → quadro normativo o atti del procedimento.",
         "Il preambolo (la composizione del collegio) e le firme vengono scartati.",
         "Un cambio di sezione apre un nuovo passaggio; la sezione entra nell'intestazione del passaggio («Sezione: "
         "Fatto») e quindi nel suo embedding.",
         "I pesi per sezione privilegiano fatti e motivazione rispetto agli atti del procedimento."),
        before=Config(V0, "bge-m3"), after=Config(V1, "bge-m3", 0.0, True),
        example="telecamere per controllare i dipendenti",
        look_for="Il primo risultato resta lo stesso, ma il passaggio scelto cambia: prima «PRESO ATTO di quanto "
                 "dichiarato…» (atti del procedimento), poi «RILEVATO che… telecamere… transito da parte dei "
                 "lavoratori» (fatti).",
        metrics=("chunks", "labelled", "chunks_per_doc"), example_doc=1601522),
    Stage(
        "duplicati", "dati", "Duplicati e boilerplate", "Provvedimenti in serie e testo ripetuto",
        "Passati ai 3.000 provvedimenti più recenti (2018–2025), due problemi: centinaia di provvedimenti quasi "
        "identici, scritti sullo stesso modello, e paragrafi standard ripetuti ovunque. Entrambi rubavano posto ai "
        "risultati.",
        ("Provvedimenti quasi identici trovati con MinHash e LSH su sequenze di 5 parole, confermati con somiglianza di "
         "Jaccard ≥ 0,7: ogni famiglia è indicizzata una volta sola (il più recente), gli altri compaiono come "
         "«decisioni analoghe».",
         "La famiglia più grande conta 694 deliberazioni quasi identiche sul revenge porn (art. 144-bis).",
         "Un paragrafo ripetuto alla lettera in almeno 25 provvedimenti è boilerplate e viene tolto: «VISTO il "
         "Regolamento (UE) 2016/679…», «VISTA la documentazione in atti», l'avviso sui mezzi di ricorso.",
         "I provvedimenti dell'era GDPR sono molto più lunghi: 13,7 passaggi ciascuno, contro 5,2."),
        before=Config(V2, "bge-m3", 0.0, True), after=Config(V3, "bge-m3", 0.0, True),
        example="revenge porn",
        look_for="«Revenge porn»: prima i primi risultati sono tutti copie della stessa deliberazione; dopo ne resta una, "
                 "con «+693 decisioni analoghe», e gli altri posti vanno a provvedimenti diversi.",
        metrics=("indexed", "families", "boilerplate", "chunks"), example_doc=9964995),
    Stage(
        "formule", "dati", "Formule procedurali", "Togliere le formule procedurali",
        "Molte frasi sono di rito: l'ingiunzione di pagamento, l'annotazione nel registro interno, la pubblicazione sul "
        "sito. Cambiano nomi e importi, quindi non sono identiche: le riconosciamo frase per frase e le togliamo.",
        ("Una sequenza di 5 parole (con numeri e nomi anonimizzati normalizzati) è procedurale se compare in almeno 25 "
         "provvedimenti e nessun argomento ne copre più della metà: il lessico tipico di un tema, come il diritto "
         "all'oblio, resta.",
         "Si toglie la frase fatta per almeno il 60% di sequenze procedurali, a meno che dica quali articoli sono stati "
         "violati («per la violazione degli artt. 5 e 13…»).",
         "Le formule pesavano soprattutto nella decisione e nei fatti."),
        before=Config(V3, "bidirlm", 0.3, True), after=Config(V4, "bidirlm", 0.3, True),
        example="pubblicazione del provvedimento sul sito del Garante",
        look_for="«Pubblicazione del provvedimento sul sito del Garante»: prima quasi tutti i passaggi trovati sono la formula finale "
                 "(«copia del presente provvedimento verrà pubblicata…»); dopo, provvedimenti che trattano davvero la "
                 "pubblicazione online di dati personali.",
        metrics=("procedural", "formula_words", "chunks"), example_doc=9964995),
    Stage(
        "sottosezioni", "dati", "Sotto-sezioni numerate", "Seguire la struttura numerata dei provvedimenti GDPR",
        "I provvedimenti recenti non usano più PREMESSO e CONSIDERATO, ma paragrafi numerati: «1. Introduzione», "
        "«3. Valutazioni dell'Autorità», «4.1 Assenza di una base giuridica». Ora la sezione si legge dai titoli e i "
        "passaggi seguono le sotto-sezioni.",
        ("Un titolo è numerato, breve e senza verbo; le sue parole decidono la sezione («valutazioni», «base "
         "giuridica» → motivazione; «ordinanza ingiunzione», «misure correttive» → decisione).",
         "Le sezioni vanno solo avanti, fatti → motivazione → decisione: un sotto-titolo dentro la motivazione («4.2 Il "
         "trattamento effettuato») non riporta ai fatti.",
         "Ogni passaggio porta il percorso dei titoli («Valutazioni > Assenza di una base giuridica (artt. 6 e 9)»), "
         "mostrato anche nei risultati.",
         "Prima gran parte della motivazione finiva tra i fatti, sotto un unico PREMESSO."),
        before=Config(V4, "bidirlm", 0.3, True, "violation"), after=Config(V5, "bidirlm", 0.3, True, "violation"),
        example="assenza di base giuridica per il trattamento dei dati sanitari",
        look_for="Con «solo violazioni», i passaggi arrivano dal paragrafo che il provvedimento dedica al tema, per "
                 "esempio «Assenza di un idoneo presupposto giuridico per il trattamento».",
        metrics=("motivazione", "fatti", "chunks"), example_doc=9964995),
    Stage(
        "archivio", "dati", "Tutto l'archivio", "Dal campione all'intero archivio",
        "Dal campione di 3.000 provvedimenti all'intero archivio scaricato, dal 1997 al 2025.",
        ("Famiglie di provvedimenti quasi identici, boilerplate e formule ricalcolati sull'intero archivio.",
         "Le regole sulle sezioni coprono sia lo stile classico (PREMESSO, CONSIDERATO) sia quello numerato.",
         "Embedding di tutti i passaggi con BidirLM su una GPU da 8 GB (circa 25 passaggi al secondo)."),
        before=Config(V5, "bidirlm", 0.3, True), after=Config(V6, "bidirlm", 0.3, True),
        example="videosorveglianza in condominio",
        look_for="«Videosorveglianza in condominio»: arrivano la decisione del 2009 «Videosorveglianza in un condominio» "
                 "e il provvedimento generale del 2004, prima fuori dal campione.",
        metrics=("documents", "period", "chunks", "families")),
    Stage(
        "inglese", "dati", "Senza traduzioni inglesi", "Lasciare fuori le traduzioni in inglese",
        "Un centinaio di provvedimenti è pubblicato solo in inglese e alcune pagine recenti riportano il provvedimento "
        "seguito dalla traduzione. Duplicavano il testo italiano e attiravano le domande in inglese verso le "
        "traduzioni.",
        ("Un paragrafo è inglese se contiene più parole funzionali inglesi che italiane.",
         "La traduzione è la coda della pagina inglese per almeno l'80% (e almeno 150 parole): una breve citazione in "
         "inglese dentro un provvedimento italiano resta.",
         "Le pagine solo in inglese sono escluse; il modello capisce comunque le domande in inglese."),
        before=Config(V6, "bidirlm", 0.3, True), after=Config(NOW, "bidirlm", 0.3, True),
        example="CCTV cameras monitoring employees",
        look_for="«CCTV cameras monitoring employees»: prima i primi due risultati sono traduzioni inglesi («Video "
                 "Surveillance»); dopo, provvedimenti italiani sulla videosorveglianza dei lavoratori.",
        metrics=("english", "testo", "chunks"), example_doc=10098477),
    Stage(
        "produzione", "produzione", "In produzione", "Dal prototipo al servizio",
        "L'indice costruito qui viene caricato su Supabase e servito da un'API di ricerca, che alimenta l'interfaccia "
        "web di Loki.",
        ("`upload_supabase.py` carica nel database Postgres i provvedimenti con il testo completo, i passaggi con i "
         "vettori e le statistiche delle parole; i file con cui l'API ordina i passaggi vanno nello Storage.",
         "**loki-backend** (FastAPI): trasforma la domanda in vettore con BidirLM e ordina in memoria tutti i passaggi "
         "con la stessa ricerca ibrida; dal database legge solo i testi dei passaggi restituiti.",
         "Endpoint: `POST /search` (con filtri per sezione, anno e tipologia), `GET /documents/{id}`, `GET /filters`, "
         "`GET /health`.",
         "**loki-ui** (React e Vite, su Vercel): ricerca, filtri, risultati, lettura del provvedimento, progetti e "
         "annotazioni.",
         "Sul piano gratuito di Supabase: database di circa 435 MB su 500, file dell'indice 180 MB su 1 GB.")),
]
# The example queries offered on every stage, the same everywhere so that one question can be followed through the
# versions: each stage's own example, in the order of the stages.
EXAMPLES = list(dict.fromkeys(s.example for s in STAGES if s.example))

# Retrieval scores computed by `python evolution.py eval`: the model comparison on v3, embeddings only, hybrid,
# and keywords only (BM25 alone does not depend on the model).
EVAL_MODELS = ("bge-m3", "jasper", "jina-v5-nano", "bidirlm")
EVAL_CONFIGS = ([Config(V3, m, a, True) for m in EVAL_MODELS for a in (0.0, 0.3)] + [Config(V3, "bge-m3", 1.0, True)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["eval"])
    ap.parse_args()
    cmd_eval(EVAL_CONFIGS)


if __name__ == "__main__":
    main()
