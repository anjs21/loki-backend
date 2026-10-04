"""
app.py - Streamlit search UI over the proof-of-concept index built by build_index.py (data/poc/).

  streamlit run app.py
"""
from __future__ import annotations

import html
import json
import re

import numpy as np
import streamlit as st

import build_index as b

st.set_page_config(page_title="Ricerca Provvedimenti Garante", page_icon="⚖️", layout="wide")

PASSAGE_CHARS = 700
SIMILAR_SHOWN = 30
MODEL_LABELS = {"jasper": "Jasper 600M", "bge-m3": "BGE-M3"}
STOPWORDS = {"della", "delle", "dello", "degli", "nella", "nelle", "sulla", "sulle", "dalla", "dalle", "come",
             "anche", "sono", "questo", "questa", "quale", "quali", "with", "from", "that", "their", "after"}


@st.cache_resource(show_spinner="Caricamento del modello di embedding...")
def get_model(key: str):
    model = b.load_model(key)
    b.encode_queries(model, key, ["riscaldamento"])  # the first encode pays one-off CUDA setup (~0.25 s)
    return model


@st.cache_resource(show_spinner="Caricamento degli embedding...")
def get_embeddings(key: str, n_chunks: int):
    emb = np.load(b.embeddings_path(key)).astype(np.float32)
    if len(emb) != n_chunks:
        st.error(f"Embedding {key} non allineati: {n_chunks} passaggi ma {len(emb)} vettori. "
                 f"Rieseguire `python build_index.py embed --model {key}`.")
        st.stop()
    return emb


@st.cache_resource(show_spinner="Caricamento dell'indice BM25...")
def get_bm25():
    try:
        return b.BM25()
    except (FileNotFoundError, RuntimeError) as e:
        st.warning(f"Ricerca per parole chiave non disponibile ({e}).")
        return None


@st.cache_resource(show_spinner="Caricamento dell'indice...")
def get_index():
    chunks = []
    for line in (b.OUT / "chunks.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        c.pop("embed_text", None)  # not needed for display; halves memory
        chunks.append(c)
    weights = np.array([c.get("weight", 1.0) for c in chunks], dtype=np.float32)
    years = np.array([int((c["date"] or "0")[:4]) for c in chunks])
    meta = json.loads((b.OUT / "chunks_meta.json").read_text(encoding="utf-8"))
    tipologie = sorted({t for c in chunks for t in c["tipologia"]})
    dup_path = b.OUT / "duplicates.json"
    similar = {int(k): v for k, v in json.loads(dup_path.read_text(encoding="utf-8")).items()} if dup_path.exists() else {}
    return chunks, weights, years, meta, tipologie, similar


def md_escape(text: str) -> str:
    """Escape Markdown in link text."""
    return re.sub(r"([\[\]*_`$])", r"\\\1", text)


def highlight(text: str, query: str) -> str:
    """HTML-escaped text with query words marked (words of 4+ letters, prefix match to catch inflections)."""
    out = html.escape(text).replace("\n", "<br>")
    words = {w.lower() for w in re.findall(r"\w{4,}", query) if w.lower() not in STOPWORDS}
    for w in sorted(words, key=len, reverse=True):
        stem = re.escape(w[:max(4, len(w) - 2)])
        out = re.sub(rf"(?i)\b({stem}\w*)", r"<mark>\1</mark>", out)
    return out


def passage(label: str, score: float, chunk: dict, query: str) -> None:
    section = b.SECTION_NAMES.get(chunk.get("section", "testo"), "Testo")
    st.markdown(f"**{label}** · <small>{section} · score {score:.3f}</small>", unsafe_allow_html=True)
    text = chunk["text"]
    short = text if len(text) <= PASSAGE_CHARS else text[:PASSAGE_CHARS].rsplit(" ", 1)[0] + " …"
    st.markdown(f"<div class='passage'>{highlight(short, query)}</div>", unsafe_allow_html=True)
    if len(text) > PASSAGE_CHARS:
        with st.expander("Passaggio completo", expanded=False):
            st.markdown(f"<div class='passage'>{highlight(text, query)}</div>", unsafe_allow_html=True)


st.markdown("""
<style>
.passage { font-size: 0.92rem; line-height: 1.5; padding: 0.5rem 0.75rem; border-left: 3px solid #9aa5b1;
           background: rgba(127,127,127,0.06); border-radius: 4px; margin-bottom: 0.5rem; }
mark { background: #ffe58f; color: inherit; padding: 0 1px; border-radius: 2px; }
.tag { display: inline-block; font-size: 0.78rem; padding: 1px 8px; margin: 0 4px 4px 0; border-radius: 10px;
       background: rgba(100,130,180,0.15); }
</style>
""", unsafe_allow_html=True)

chunks, weights, years, meta, tipologie, similar = get_index()
available = [m for m in b.MODELS if b.embeddings_path(m).exists()]
if not available:
    st.error("Nessun embedding trovato: eseguire `python build_index.py embed`.")
    st.stop()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Ricerca")
    model_key = st.selectbox("Modello di embedding", available,
                             index=available.index(b.DEFAULT_MODEL) if b.DEFAULT_MODEL in available else 0,
                             format_func=lambda m: MODEL_LABELS.get(m, m))
    alpha = st.slider("Peso parole chiave (BM25)", 0.0, 1.0, b.HYBRID_ALPHA, step=0.1,
                      help="Ricerca ibrida: fonde la classifica semantica e quella per parole chiave (BM25) "
                           "con reciprocal rank fusion. 0 = solo semantica, 1 = solo parole chiave.")
    st.header("Filtri")
    y0, y1 = int(years.min()), int(years.max())
    year_range = st.slider("Anno", y0, y1, (y0, y1)) if y0 < y1 else (y0, y1)
    tip_sel = st.multiselect("Tipologia", tipologie, placeholder="Tutte")
    k = st.slider("Numero di risultati", 5, 50, 10, step=5)
    use_weights = st.toggle("Privilegia fatti e motivazione", value=True,
                            help="Pesa i passaggi per sezione: fatti/motivazione 1.0, decisione e quadro normativo "
                                 "0.97, atti del procedimento 0.9. Disattivare per la similarità pura.")
    st.divider()
    n_docs = meta.get("indexed_documents", meta["documents"])
    st.caption(f"Indice: {n_docs:,} provvedimenti distinti ({meta['documents']:,} con i duplicati) · "
               f"{meta['chunks']:,} passaggi\n\nPeriodo: {meta['date_range'][0]} → {meta['date_range'][1]}")

# Load the selected model when the page opens rather than on the first search; cached for the life of the server.
model = get_model(model_key)
emb = get_embeddings(model_key, len(chunks))
bm25 = get_bm25() if alpha > 0 else None

# ----------------------------------------------------------------------------- search
st.title("Ricerca nei provvedimenti del Garante")
st.caption("Ricerca ibrida (semantica + parole chiave, anche in inglese) su attività di trattamento e violazioni "
           "contestate.")

query = st.text_input("Cerca", placeholder="es. telecamere per controllare i dipendenti, data breach ransomware, "
                                           "telemarketing registro delle opposizioni", label_visibility="collapsed")

if query.strip():
    with st.spinner("Ricerca in corso..."):
        q = b.encode_queries(model, model_key, [query])[0]
        scores = b.score_chunks(q, query, emb, bm25, weights if use_weights else None, alpha)
        mask = (years >= year_range[0]) & (years <= year_range[1])
        if tip_sel:
            sel = set(tip_sel)
            mask &= np.array([bool(sel.intersection(c["tipologia"])) for c in chunks])
        results = b.rank_documents(scores, chunks, k, mask=mask)

    if not results:
        st.info("Nessun risultato con i filtri selezionati.")
    for rank, (doc_id, hits) in enumerate(results.items(), 1):
        top_score, top_i = hits["top"]
        c = chunks[top_i]
        with st.container(border=True):
            left, right = st.columns([0.85, 0.15])
            left.markdown(f"**{rank}. [{md_escape(c['title'])}]({c['url']})**")
            score_label = "coseno" if bm25 is None else "BM25" if alpha >= 1 else "RRF"
            right.markdown(f"<div style='text-align:right'><small>{score_label}</small><br><b>{top_score:.4g}</b></div>",
                           unsafe_allow_html=True)
            tags = [f"<span class='tag'>📅 {c['date']}</span>"]
            tags += [f"<span class='tag'>{html.escape(t)}</span>" for t in c["tipologia"]]
            tags += [f"<span class='tag'>#{html.escape(a)}</span>" for a in c["argomenti"][:6]]
            twins = similar.get(doc_id, [])
            if twins:
                tags.append(f"<span class='tag'>+{len(twins)} decisioni analoghe</span>")
            st.markdown(" ".join(tags), unsafe_allow_html=True)
            shown = [r for r in ("Attività", "Violazione") if r in hits]
            if shown:
                cols = st.columns(len(shown))
                for col, role in zip(cols, shown):
                    with col:
                        s, i = hits[role]
                        passage(role, s, chunks[i], query)
            else:
                passage("Passaggio più rilevante", top_score, c, query)
            if twins:
                with st.expander(f"{len(twins)} decisioni con testo quasi identico (stesso modello di provvedimento)"):
                    st.markdown("\n".join(f"- {t['date']} · [{md_escape(t['title'])}]({t['url']})"
                                           for t in twins[:SIMILAR_SHOWN]))
                    if len(twins) > SIMILAR_SHOWN:
                        st.caption(f"... e altre {len(twins) - SIMILAR_SHOWN}.")
else:
    st.info("Digita una ricerca per iniziare. Puoi filtrare per anno e tipologia dalla barra laterale.")
