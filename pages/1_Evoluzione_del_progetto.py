"""
Evoluzione del progetto - how the search evolved, stage by stage, with live before/after comparisons run on the
snapshots of each version of the index (see evolution.py). A page of the app: `streamlit run app.py`.

The retrieval scores in the charts come from `python evolution.py eval` (data/poc/evolution_eval.json).
"""
from __future__ import annotations

import dataclasses
import html
import json
import re
import urllib.parse

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import build_index as b
import eval_retrieval as ev
import evolution as e
import phase_maps as pm
from search_ui import (IT_AXIS, STYLE, chart_colors, dec, get_model, highlight, html_md, inline_md, num,
                       query_words)

st.set_page_config(page_title="Evoluzione del progetto", page_icon="📈", layout="wide")
st.markdown(STYLE + """
<style>
.lead { font-size: 1.15rem; line-height: 1.55; margin: 0.25rem 0 1rem; }
.ev-sec { font-size: 0.82rem; opacity: 0.75; margin: 0.15rem 0 0.25rem; }
</style>
""", unsafe_allow_html=True)

SNIPPET_CHARS = 360
MAPS_PAGE = "Mappe_degli_embedding"  # the URL of pages/2_Mappe_degli_embedding.py
MAP_VIEW = {"duplicati": {"proiezione": "pca"}}  # where a stage's change shows best on the map
EVAL_TAGS = dict(ev.QUERIES)  # evaluation query -> the topic tags that make a decision relevant
# Chart colours from the reference palette (accent, second series, de-emphasis, ink), stepped for light or dark.
_colors = chart_colors()
ACCENT, SECOND, MUTED_MARK, INK, INK_MUTED = (_colors[k] for k in ("accent", "second", "muted_mark", "ink", "ink_muted"))


# ----------------------------------------------------------------------------- formatting
def link(title: str, url: str) -> str:
    return f"<a href='{html.escape(url)}' target='_blank'>{html.escape(title)}</a>"


def signed(d) -> str | None:
    if not d:
        return None
    return ("+" if d > 0 else "-") + num(abs(d))


def section_label(section: str, heading: str = "") -> str:
    label = "Senza sezione" if section == "testo" else b.SECTION_NAMES.get(section, section)
    if heading:
        label += " · " + html.escape(heading.split(" > ")[-1][:110])
    return label


def snippet(text: str, query: str) -> str:
    """The start of the passage, or the stretch around the first query word when that comes later."""
    if len(text) <= SNIPPET_CHARS:
        return highlight(text, query)
    start = 0
    words = query_words(query)
    if words:
        m = re.search(r"(?i)\b(" + "|".join(re.escape(w[:max(4, len(w) - 2)]) for w in words) + ")", text)
        if m and m.start() > SNIPPET_CHARS * 0.6:
            start = text.rfind(" ", 0, max(0, m.start() - 80)) + 1
    end = start + SNIPPET_CHARS
    piece = text[start:end].rsplit(" ", 1)[0] if end < len(text) else text[start:]
    return ("… " if start else "") + highlight(piece, query) + (" …" if end < len(text) else "")


# ----------------------------------------------------------------------------- data
@st.cache_resource(show_spinner="Apertura della versione dell'indice...", max_entries=10)
def get_snapshot(name: str) -> e.Snapshot:
    return e.open_snapshot(name)


@st.cache_resource(show_spinner="Caricamento dell'indice BM25...", max_entries=10)
def get_bm25(name: str):
    return e.open_bm25(get_snapshot(name))


@st.cache_data(show_spinner=False)
def read_meta(name: str) -> dict:
    return e.read_meta(name)


@st.cache_data(show_spinner=False, max_entries=1000)
def query_vector(model: str, query: str) -> np.ndarray:
    return b.encode_queries(get_model(model), model, [query])[0]


@st.cache_data(show_spinner=False, max_entries=1000)
def run(cfg: e.Config, query: str, k: int) -> list[dict]:
    snap = get_snapshot(cfg.snapshot)
    bm25 = get_bm25(cfg.snapshot) if cfg.alpha > 0 else None
    out = []
    for d, hits in e.search(cfg, snap, bm25, query_vector(cfg.model, query), query, k).items():
        i = hits["top"][1]
        c = snap.chunks[i]
        out.append({**snap.doc(i), "section": c["section"], "heading": c["heading"], "text": snap.text(i),
                    "similar": len(snap.similar.get(d, []))})
    return out


@st.cache_data(show_spinner=False)
def doc_passages(name: str, docweb_id: int) -> list[dict]:
    snap = get_snapshot(name)
    return [{"section": c["section"], "heading": c["heading"], "text": snap.text(i)}
            for i, c in enumerate(snap.chunks) if c["docweb_id"] == docweb_id]


@st.cache_data(show_spinner=False)
def scope_counts(name: str, scope: str | None) -> tuple[int, int]:
    """Passages and decisions a search covers in that scope."""
    snap = get_snapshot(name)
    rows = np.flatnonzero(b.scope_mask(snap.chunks, scope))
    return len(rows), len({snap.chunks[i]["docweb_id"] for i in rows})


@st.cache_data(show_spinner=False)
def load_eval(mtime: float) -> dict:
    return json.loads(e.EVAL_PATH.read_text(encoding="utf-8"))


def eval_results() -> dict:
    return load_eval(e.EVAL_PATH.stat().st_mtime) if e.EVAL_PATH.exists() else {}


def warm_up() -> None:
    """Load every model and snapshot and run every example query, so the presentation runs without waiting."""
    jobs = [(cfg, q) for s in e.STAGES for q in e.EXAMPLES
            for cfg in [s.before, s.after] + [dataclasses.replace(s.after, model=m) for m in s.alternatives] if cfg]
    jobs.sort(key=lambda j: j[0].model)  # one model after another
    bar = st.progress(0.0, text="Preparazione...")
    for n, (cfg, q) in enumerate(jobs, 1):
        bar.progress(n / len(jobs), text=f"{b.MODELS[cfg.model]['label']} · {e.SNAPSHOT_LABELS[cfg.snapshot]} · {q}")
        run(cfg, q, st.session_state.get("k", 5))
    bar.empty()
    st.toast("Tutte le tappe sono pronte.")


# ----------------------------------------------------------------------------- pieces
def metric_row(stage: e.Stage) -> None:
    cards = []  # (label, value, delta, help)
    after = read_meta(stage.after.snapshot) if stage.after else None
    before = read_meta(stage.before.snapshot) if stage.before else None
    for key in stage.metrics:
        label, fn = e.METRICS[key]
        a = fn(after)
        if before is None:
            cards.append((label, a if isinstance(a, str) else num(a), None, None))
        else:
            p = fn(before)
            if isinstance(a, str):
                cards.append((label, a, None, f"Prima: {p}"))
            else:
                cards.append((label, num(a), signed(a - p), f"Prima: {num(p)}"))
    if stage.before and stage.after and stage.after.scope and stage.before.scope != stage.after.scope:
        (c0, d0), (c1, d1) = scope_counts(stage.before.snapshot, stage.before.scope), \
            scope_counts(stage.after.snapshot, stage.after.scope)
        cards += [("Passaggi in cui si cerca", num(c1), signed(c1 - c0), f"Prima: {num(c0)}"),
                  ("Provvedimenti in cui si cerca", num(d1), signed(d1 - d0), f"Prima: {num(d0)}")]
    if cards:
        for col, (label, value, delta, help_) in zip(st.columns(len(cards)), cards):
            col.metric(label, value, delta, delta_color="off", help=help_, border=True)


def result_card(rank: int, r: dict, query: str) -> None:
    with st.container(border=True):
        html_md(f"<p><b>{rank}. {link(r['title'], r['url'])}</b></p>")
        tags = [f"<span class='tag'>📅 {r['date']}</span>"]
        tags += [f"<span class='tag'>{html.escape(t)}</span>" for t in r["tipologia"][:1]]
        if r["similar"]:
            tags.append(f"<span class='tag'>+{r['similar']} decisioni analoghe</span>")
        if query in EVAL_TAGS and EVAL_TAGS[query] & set(r["argomenti"]):
            tags.append("<span class='tag'>✓ argomento atteso</span>")
        html_md(f"<div>{''.join(tags)}</div>")
        html_md(f"<div class='ev-sec'>{section_label(r['section'], r['heading'])}</div>"
                f"<div class='passage'>{snippet(r['text'], query)}</div>")


def results_column(label: str, cfg: e.Config, query: str, k: int) -> None:
    st.markdown(f"#### {label}")
    st.caption(f"Indice {e.SNAPSHOT_LABELS[cfg.snapshot]} · {cfg.describe()}")
    eff = b.effective_alpha(query, cfg.alpha)
    if cfg.alpha > 0 and eff != cfg.alpha:
        st.caption(f"Riferimenti esatti nella domanda (numeri o virgolette): peso parole chiave {dec(eff)}.")
    if query in EVAL_TAGS:
        res = eval_results().get(e.config_key(cfg))
        if res:
            i = [q for q, _ in ev.QUERIES].index(query)
            st.caption(f"Domanda della valutazione · P@10 {dec(res['per_query'][i][0])}")
    with st.spinner("Ricerca in corso..."):
        rows = run(cfg, query, k)
    if not rows:
        st.info("Nessun risultato.")
    for rank, r in enumerate(rows, 1):
        result_card(rank, r, query)


def live_comparison(stage: e.Stage, k: int) -> e.Config:
    """The before/after results; returns the configuration shown as «dopo» (the model can be switched)."""
    st.subheader("Prova dal vivo")
    if stage.note:
        st.caption(stage.note)
    qkey = f"q_{stage.key}"
    # the same examples on every stage; each stage opens on its own
    preset = st.pills("Esempi", e.EXAMPLES, default=stage.example, key=f"pill_{stage.key}",
                      on_change=lambda: st.session_state.update({qkey: ""}))
    custom = st.columns([2, 3])[0].text_input("Oppure una ricerca libera", key=qkey,
                                              placeholder="Scrivi e premi Invio")
    query = custom.strip() or preset or stage.example
    after = stage.after
    if stage.alternatives:
        model = st.segmented_control("Modello del lato «dopo»", stage.alternatives, default=stage.after.model,
                                     format_func=lambda m: b.MODELS[m]["label"], key=f"alt_{stage.key}")
        after = dataclasses.replace(stage.after, model=model or stage.after.model)
    if stage.look_for and query == stage.example:
        st.info(stage.look_for, icon=":material/visibility:")
    elif stage.example:
        st.caption(f"L'esempio di questa tappa è «{stage.example}».")
    if stage.before:
        cols = st.columns(2, gap="large")
        with cols[0]:
            results_column("Prima", stage.before, query, k)
        with cols[1]:
            results_column("Dopo", after, query, k)
    else:
        col, _ = st.columns([3, 2])
        with col:
            results_column("Allora", after, query, k)
    return after


def example_panel(stage: e.Stage) -> None:
    d = stage.example_doc
    doc = get_snapshot(stage.after.snapshot).docs.get(d) or get_snapshot(stage.before.snapshot).docs.get(d)
    st.subheader("Lo stesso provvedimento, prima e dopo")
    html_md(f"<p>{link(doc['title'], doc['url'])} · {doc['date']} · {html.escape(', '.join(doc['tipologia']))}</p>")
    for col, label, cfg in zip(st.columns(2, gap="large"), ("Prima", "Dopo"), (stage.before, stage.after)):
        passages = doc_passages(cfg.snapshot, d)
        with col:
            chars = sum(len(p["text"]) for p in passages)
            html_md(f"<p><b>{label}</b> · indice {html.escape(e.SNAPSHOT_LABELS[cfg.snapshot])}: {len(passages)} "
                    f"passaggi, {num(chars)} caratteri</p>")
            with st.container(height=520):
                if not passages:
                    st.caption("Non presente in questa versione.")
                for n, p in enumerate(passages, 1):
                    html_md(f"<div class='ev-sec'><b>Passaggio {n}</b> · {section_label(p['section'], p['heading'])}"
                            f"</div><div class='passage'>{html.escape(p['text']).replace(chr(10), '<br>')}</div>")


def chart_bm25() -> None:
    res = eval_results()
    methods = [("Solo semantica", 0.0), ("Solo parole chiave", 1.0), ("Ibrida", 0.3)]
    langs = ["Domande in italiano (15)", "Domande in inglese (7)"]
    rows = []
    for name, alpha in methods:
        r = res.get(e.config_key(e.Config(e.V3, "bge-m3", alpha, True)))
        if not r:
            st.caption("Punteggi non calcolati: eseguire `python evolution.py eval`.")
            return
        rows += [{"Ricerca": name, "Domande": langs[0], "P@10": r["it"][0]},
                 {"Ricerca": name, "Domande": langs[1], "P@10": r["en"][0]}]
    df = pd.DataFrame(rows)
    df["valore"] = df["P@10"].map(dec)
    enc = dict(x=alt.X("Ricerca:N", sort=[m for m, _ in methods], title=None,
                       axis=alt.Axis(labelAngle=0, labelFontSize=12, domainColor=INK_MUTED, ticks=False)),
               xOffset=alt.XOffset("Domande:N", sort=langs, scale=alt.Scale(paddingInner=0.06)),
               y=alt.Y("P@10:Q", scale=alt.Scale(domain=[0, 1]), title="P@10",
                       axis=alt.Axis(tickCount=5, labelExpr=IT_AXIS, gridOpacity=0.4, domain=False)))
    bars = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        **enc, color=alt.Color("Domande:N", sort=langs, scale=alt.Scale(domain=langs, range=[ACCENT, SECOND]),
                               legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
        tooltip=[alt.Tooltip("Ricerca:N"), alt.Tooltip("Domande:N"), alt.Tooltip("valore:N", title="P@10")])
    labels = alt.Chart(df).mark_text(dy=-8, fontSize=12, color=INK).encode(**enc, text="valore:N")
    html_md("<p><b>Quota di provvedimenti pertinenti nei primi 10</b> · BGE-M3, indice v3</p>")
    st.altair_chart((bars + labels).properties(height=280), width="stretch")


def chart_models(selected: str) -> None:
    res = eval_results()
    rows = []
    for m in e.EVAL_MODELS:
        dense, hybrid = (res.get(e.config_key(e.Config(e.V3, m, a, True))) for a in (0.0, 0.3))
        if not dense or not hybrid:
            st.caption("Punteggi non calcolati: eseguire `python evolution.py eval`.")
            return
        rows.append({"key": m, "Modello": b.MODELS[m]["label"], "P@10": hybrid["all"][0],
                     "valore": dec(hybrid["all"][0]), "MRR": dec(hybrid["all"][1]),
                     "P@10 italiano": dec(hybrid["it"][0]), "P@10 inglese": dec(hybrid["en"][0]),
                     "P@10 solo semantica": dec(dense["all"][0])})
    df = pd.DataFrame(rows)
    enc = dict(y=alt.Y("Modello:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=12, ticks=False, domain=False)),
               x=alt.X("P@10:Q", scale=alt.Scale(domain=[0, 1]), title="P@10",
                       axis=alt.Axis(tickCount=5, labelExpr=IT_AXIS, gridOpacity=0.4, domain=False)))
    bars = alt.Chart(df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=22).encode(
        **enc, color=alt.condition(alt.datum.key == selected, alt.value(ACCENT), alt.value(MUTED_MARK)),
        tooltip=[alt.Tooltip("Modello:N"), alt.Tooltip("valore:N", title="P@10 (ibrida)"),
                 alt.Tooltip("MRR:N", title="MRR (ibrida)"), alt.Tooltip("P@10 italiano:N"),
                 alt.Tooltip("P@10 inglese:N"), alt.Tooltip("P@10 solo semantica:N")])
    labels = alt.Chart(df).mark_text(align="left", dx=6, fontSize=12, color=INK).encode(**enc, text="valore:N")
    html_md(f"<p><b>P@10 con ricerca ibrida, 22 domande</b> · in evidenza: {b.MODELS[selected]['label']}</p>")
    st.altair_chart((bars + labels).properties(height=200), width="stretch")
    with st.expander("Tutti i punteggi"):
        st.dataframe(df[["Modello", "valore", "MRR", "P@10 italiano", "P@10 inglese", "P@10 solo semantica"]]
                     .rename(columns={"valore": "P@10"}), hide_index=True, width="stretch")


def architecture() -> None:
    fg, line = INK, INK_MUTED
    st.graphviz_chart(f"""digraph {{
  rankdir=LR; bgcolor="transparent"; nodesep=0.4; ranksep=0.6; fontsize=14;
  node [shape=box, style=rounded, fontname="Helvetica", fontsize=14, margin="0.2,0.1", color="{line}", fontcolor="{fg}"];
  edge [color="{line}", fontcolor="{fg}", fontname="Helvetica", fontsize=12];
  subgraph cluster_loki {{ label="repository loki"; fontname="Helvetica"; fontcolor="{fg}"; color="{line}";
    style="rounded,dashed";
    scraper [label="garante_scraper.py\\nscarica i provvedimenti"];
    build [label="build_index.py\\npassaggi, embedding, BM25"];
    upload [label="upload_supabase.py"];
    scraper -> build -> upload; }}
  subgraph cluster_sb {{ label="Supabase"; fontname="Helvetica"; fontcolor="{fg}"; color="{line}"; style="rounded";
    db [label="Postgres\\nprovvedimenti, passaggi, vettori"];
    storage [label="Storage\\nfile dell'indice"]; }}
  api [label="loki-backend · FastAPI\\nembedding della domanda\\nricerca ibrida in memoria"];
  ui [label="loki-ui · React\\nsu Vercel"];
  upload -> db; upload -> storage;
  storage -> api [label="alla build"];
  api -> db [label="testi dei passaggi"];
  ui -> api [label="HTTPS"];
}}""", width="stretch")


# ----------------------------------------------------------------------------- pages
def overview() -> None:
    st.title("Evoluzione del progetto")
    html_md("<p class='lead'>Loki cerca nei provvedimenti del Garante per la protezione dei dati personali. Il "
                "prototipo è passato da una ricerca semantica su un campione a una ricerca ibrida sull'intero "
                "archivio, capace di cercare soltanto nelle violazioni contestate. Due binari: <b>come cerchiamo</b> e "
                "<b>cosa indicizziamo</b>.</p>")
    st.subheader("Le tappe")
    n = 0
    for part, name in e.PARTS.items():
        stages = [s for s in e.STAGES if s.part == part]
        lines = [f"**{name}**", "", "| | Tappa | In breve |", "|---|---|---|"]
        for s in stages:
            n += 1
            lines.append(f"| {n} | **{s.title}** | {s.summary} |")
        st.markdown("\n".join(lines))
    st.caption("Ogni tappa confronta dal vivo due versioni reali dell'indice, salvate nella cartella data/poc al momento del "
               "cambiamento: tra «prima» e «dopo» cambia una cosa sola.")
    with st.expander("Le versioni dell'indice"):
        rows = []
        for name, label in e.SNAPSHOT_LABELS.items():
            m = read_meta(name)
            rows.append({"Versione": label, "Periodo": e.METRICS["period"][1](m), "Provvedimenti": num(m["documents"]),
                         "Indicizzati": num(e.METRICS["indexed"][1](m)), "Passaggi": num(m["chunks"]),
                         "Passaggi per provvedimento": num(float(m["chunks_per_doc"])),
                         "Modelli": ", ".join(b.MODELS[k]["label"] for k in b.MODELS
                                              if (((b.OUT / name) if name else b.OUT) / b.MODELS[k]["file"]).exists())})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.page_link("pages/2_Mappe_degli_embedding.py", label="Le mappe degli embedding di ogni versione",
                 icon=":material/scatter_plot:")


def map_links(stage: e.Stage) -> None:
    """Links to the embedding maps of the versions the stage compares (pages/2_Mappe_degli_embedding.py)."""
    links: dict[str, list[str]] = {}
    for side, cfg in (("prima", stage.before), ("dopo", stage.after)):
        if cfg is None or not (pm.MAPS / f"{pm.map_key(cfg.snapshot, cfg.model, cfg.scope)}.npz").exists():
            continue
        version = e.SNAPSHOT_LABELS[cfg.snapshot].split(" · ")[0]
        query = {"versione": version, "modello": cfg.model, **({"passaggi": "violazioni"} if cfg.scope else {}),
                 **MAP_VIEW.get(stage.key, {})}
        label = f"{version} · {b.MODELS[cfg.model]['label']}" + (" · solo violazioni" if cfg.scope else "")
        links.setdefault(f"{MAPS_PAGE}?{urllib.parse.urlencode(query)}", [label]).append(side)
    if links:
        items = [f"{'/'.join(sides) + ': ' if len(links) > 1 else ''}<a href='{html.escape(href)}' target='_self'>"
                 f"{html.escape(label)}</a>" for href, (label, *sides) in links.items()]
        html_md("<p class='ev-sec'>Mappe degli embedding · " + " · ".join(items) + "</p>")


def stage_page(n: int, stage: e.Stage, k: int) -> None:
    st.caption(f"{e.PARTS[stage.part]} · tappa {n} di {len(e.STAGES)}")
    st.title(stage.title)
    html_md(f"<p class='lead'>{html.escape(stage.summary, quote=False)}</p>")
    metric_row(stage)
    bullets = "<ul>" + "".join(f"<li>{inline_md(p)}</li>" for p in stage.points) + "</ul>"
    if stage.key in ("bm25", "modelli"):  # the scores beside the text
        left, chart_box = st.columns(2, gap="large")
        html_md(bullets, left)
    else:
        html_md(bullets)
    map_links(stage)
    if stage.key == "produzione":
        architecture()
        return
    after = live_comparison(stage, k)
    if stage.key == "bm25":
        with chart_box:
            chart_bm25()
    elif stage.key == "modelli":
        with chart_box:
            chart_models(after.model)
    if stage.example_doc:
        st.divider()
        example_panel(stage)


# ----------------------------------------------------------------------------- navigation
labels = ["Panoramica"] + [f"{i}\\. {s.short}" for i, s in enumerate(e.STAGES, 1)]  # "\\." or Markdown reads a list
if "tappa" not in st.session_state:  # the stage is in the address (?tappa=3), so a reload or a link keeps it
    try:
        st.session_state.tappa = min(max(int(st.query_params.get("tappa", 0)), 0), len(labels) - 1)
    except ValueError:
        st.session_state.tappa = 0


def go(step: int) -> None:
    st.session_state.tappa = min(max(st.session_state.tappa + step, 0), len(labels) - 1)


with st.sidebar:
    st.header("Tappe")
    st.radio("Tappa", range(len(labels)), format_func=labels.__getitem__, key="tappa", label_visibility="collapsed")
    ranges = []
    for part, name in e.PARTS.items():
        nums = [i for i, s in enumerate(e.STAGES, 1) if s.part == part]
        ranges.append(f"{nums[0]}–{nums[-1]} {name.lower()}" if len(nums) > 1 else f"{nums[0]} {name.lower()}")
    st.caption(" · ".join(ranges))
    st.divider()
    k = st.slider("Risultati per colonna", 3, 10, 5, key="k")
    if st.button("Prepara tutte le tappe", help="Carica tutti i modelli e le versioni dell'indice ed esegue gli "
                                                "esempi, così durante la presentazione non si aspetta (1–2 minuti)."):
        warm_up()
    st.caption("Ogni modello si carica al primo uso (10–30 s), poi resta in memoria.")

idx = st.session_state.tappa
st.query_params["tappa"] = str(idx)
if idx == 0:
    overview()
else:
    stage_page(idx, e.STAGES[idx - 1], k)

st.divider()
prev_col, _, next_col = st.columns([2, 3, 2])
if idx > 0:
    prev_col.button(f"← {labels[idx - 1]}", on_click=go, args=(-1,), width="stretch")
if idx < len(labels) - 1:
    next_col.button(f"{labels[idx + 1]} →", on_click=go, args=(1,), type="primary", width="stretch")
