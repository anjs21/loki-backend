"""
Mappe degli embedding - every version of the index as a map: each dot a decision, placed by its embedding (projected
to 2D with PCA or t-SNE), to see what each change did to the space the search works in. A page of the app:
`streamlit run app.py`. The maps come from `python phase_maps.py` (data/poc/maps/).
"""
from __future__ import annotations

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import build_index as b
import evolution as e
import phase_maps as pm
from search_ui import chart_colors, dec, html_md, num

st.set_page_config(page_title="Mappe degli embedding", page_icon="🗺️", layout="wide")
st.markdown("""
<style>
.lead { font-size: 1.15rem; line-height: 1.55; margin: 0.25rem 0 1rem; }
.nn { font-size: 0.9rem; line-height: 1.45; padding-left: 1.1rem; }
.nn li { margin-bottom: 0.35rem; }
.nn small, .doc small { opacity: 0.75; }
</style>
""", unsafe_allow_html=True)

C = chart_colors()
TYPE_GROUPS = ("Decisioni su casi", "Pareri", "Deliberazioni e altri atti")
TYPE_COLORS = (C["accent"], C["second"], C["third"])
# Sequential ramp for the year, light to dark on a light page and dark to light on a dark one (reference palette).
YEAR_RAMP = ["#184f95", "#b7d3f6"] if C["dark"] else ["#9ec5f4", "#0d366b"]
PROJECTIONS = {"tsne": "t-SNE", "pca": "PCA"}
SCOPE_LABELS = {None: "Tutto il provvedimento", "violation": "Solo violazioni"}
COLOR_MODES = ("Tipologia", "Anno", "Argomento")
DEFAULT_TOPIC = "Videosorveglianza"

# What to look at on each version's map (whole decisions, the model of that version).
NOTES = {
    e.V0: "3.000 provvedimenti dal 2004 al 2018, quasi tutti decisioni su ricorso (in blu). Pareri e altri atti "
          "(arancio e verde) stanno già in una zona a parte: il modello riconosce il tipo di documento.",
    e.V1: "Gli stessi 3.000 provvedimenti, divisi in passaggi per sezione (fatti, motivazione, decisione). La mappa "
          "per provvedimento cambia poco; da qui però si può guardare solo la parte sulle violazioni: "
          "«Passaggi → Solo violazioni».",
    e.V2: "Provvedimenti dal 2018 al 2025. Con la proiezione PCA, le 694 deliberazioni quasi identiche sul revenge "
          "porn (in verde) sono un blocco compatto lontano da tutto il resto; t-SNE lo spezza in tante isole verdi. "
          "Nella ricerca, quelle copie riempivano i primi risultati.",
    e.V3: "Ogni famiglia di provvedimenti quasi identici è ora un solo punto, il suo rappresentante: il blocco verde "
          "di v2 non c'è più. Restano 2.224 provvedimenti. Qui ci sono le mappe di tutti e quattro i modelli messi a "
          "confronto.",
    e.V4: "Tolte le formule procedurali, la mappa per provvedimento cambia poco (AUC citazioni 0,93, come in v3): le "
          "formule pesano soprattutto sui singoli passaggi, cioè su ciò che la ricerca mette in fila.",
    e.V5: "I passaggi seguono la numerazione interna dei provvedimenti GDPR. Con «Solo violazioni» i provvedimenti che "
          "si citano si avvicinano: AUC da 0,81 (v4) a 0,91, recall@10 dal 19% al 26%.",
    e.V6: "Tutto l'archivio: 10.569 provvedimenti dal 1997 al 2025. Con «Colori → Anno» si vede che prima e dopo il "
          "2018 occupano zone diverse (9 vicini su 10 sono della stessa epoca). Al centro della mappa t-SNE, un "
          "gruppetto compatto: le traduzioni inglesi, vicine tra loro (6 dei loro 10 vicini sono altre traduzioni).",
    e.NOW: "L'indice in produzione: come v6, senza le traduzioni inglesi. Pareri (arancio) e deliberazioni (verde) "
           "restano in zone proprie; le decisioni su casi si dividono per tema: prova «Colori → Argomento».",
}


def short(snapshot: str) -> str:
    return e.SNAPSHOT_LABELS[snapshot].split(" · ")[0]


def type_group(tipologia: list[str]) -> str:
    if not b.CASE_TYPES.isdisjoint(tipologia):
        return TYPE_GROUPS[0]
    return TYPE_GROUPS[1] if "Parere del Garante" in tipologia else TYPE_GROUPS[2]


def pick(label: str, options: list, default, key: str, format_func=str, disabled: bool = False):
    """A segmented control that always has a value: a click on the chosen option leaves it chosen, and a choice not
    offered here (another version has other models) falls back to the last valid one or the default."""
    last = key + "_last"

    def keep() -> None:
        if st.session_state[key] is None:
            st.session_state[key] = st.session_state.get(last, default)

    value = st.session_state.get(key)
    if value not in options:
        value = st.session_state.get(last) if st.session_state.get(last) in options else default
    # set on every run: when the options change (another version), Streamlit takes it for a new widget, which would
    # otherwise start with nothing chosen
    st.session_state[key] = value
    # a width of its own, from the labels: in a row, "content" squeezes the groups and their buttons wrap
    width = sum(len(format_func(o)) * 7 + 40 for o in options) + 4
    value = st.segmented_control(label, options, format_func=format_func, key=key, on_change=keep, disabled=disabled,
                                 width=width)
    st.session_state[last] = value
    return value


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def read_index() -> dict:
    return pm.read_index()


@st.cache_data(show_spinner="Caricamento della mappa...", max_entries=40)
def load_map(key: str, snapshot: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The decisions of one map with their coordinates in both projections, and their nearest decisions."""
    m = pm.read_map(key)
    docs = pm.read_docs(snapshot)
    rows = [docs[int(i)] for i in m["ids"]]
    df = pd.DataFrame({
        "id": m["ids"], "pca_x": m["pca"][:, 0], "pca_y": m["pca"][:, 1],
        "tsne_x": m["tsne"][:, 0], "tsne_y": m["tsne"][:, 1],
        "titolo": [d["t"] for d in rows], "data": [d["d"] for d in rows],
        "anno": [int((d["d"] or "0")[:4]) for d in rows], "url": [d["u"] for d in rows],
        "gruppo": [type_group(d["ty"]) for d in rows], "tipologia": [", ".join(d["ty"]) for d in rows],
        "argomenti": [d["a"] for d in rows], "simili": [d["n"] for d in rows],
    })
    df["argomenti_testo"] = [", ".join(a) if a else "—" for a in df["argomenti"]]
    return df, m["nn_idx"], m["nn_cos"].astype(np.float32)


# ----------------------------------------------------------------------------- the map
# Inside the chart: the clicked decision is in the selection's store, with the list of its nearest ("|3|17|...|").
# Looked up by field name: Streamlit adds the chart's encodings (colour, x, y) to the fields of a point selection.
_STORE = "data('doc_store')[0]"
CHOSEN_LIST = f"{_STORE}.values[indexof(pluck({_STORE}.fields, 'field'), 'vicini')]"
IS_NEAR = f"length(data('doc_store')) > 0 && indexof('' + {CHOSEN_LIST}, '|' + datum.i + '|') >= 0"
ANY_CHOSEN = "length(data('doc_store')) > 0"


def map_chart(df: pd.DataFrame, proj: str, mode: str, topic: str | None, nn_idx: np.ndarray) -> alt.Chart:
    """The map. A click sets apart the decision and its 10 nearest in the browser, through conditions on the
    selection, so the chart itself stays the same and keeps its zoom (Streamlit redraws a chart whose data change).
    One view only: Streamlit takes clicks only on single-view charts."""
    data = df.drop(columns=["argomenti"]).copy()
    data["x"], data["y"] = data[f"{proj}_x"], data[f"{proj}_y"]
    data["i"] = np.arange(len(data))
    data["vicini"] = ["|" + "|".join(map(str, row)) + "|" for row in nn_idx]
    if mode == "Argomento":
        data["evidenza"] = np.where([topic in a for a in df["argomenti"]], topic, "Altri")
    base = 14 if len(data) > 6000 else 26

    if mode == "Tipologia":
        color = alt.Color("gruppo:N", scale=alt.Scale(domain=list(TYPE_GROUPS), range=list(TYPE_COLORS)),
                          legend=alt.Legend(orient="top", title=None, labelFontSize=12, symbolSize=120))
    elif mode == "Anno":
        color = alt.Color("anno:Q", scale=alt.Scale(range=YEAR_RAMP),
                          legend=alt.Legend(orient="top", title="Anno", format="d", gradientLength=220,
                                            labelFontSize=12))
    else:
        color = alt.Color("evidenza:N", scale=alt.Scale(domain=[topic, "Altri"], range=[C["accent"], C["muted_mark"]]),
                          legend=alt.Legend(orient="top", title=None, labelFontSize=12, symbolSize=120))
    color.condition = {"param": "doc", "empty": False, "value": C["ink"]}
    chosen = {"param": "doc", "empty": False}

    def value(default, if_chosen, if_near, if_other=None):
        conditions = [{**chosen, "value": if_chosen}, {"test": IS_NEAR, "value": if_near}]
        if if_other is not None:
            conditions.append({"test": ANY_CHOSEN, "value": if_other})
        return {"value": default, "condition": conditions}

    # drawing order: the chosen one on top, then its nearest, then (by topic) the highlighted ones
    order = value(0, 3, 2)
    if mode == "Argomento":
        order["condition"].append({"test": "datum.evidenza != 'Altri'", "value": 1})
    tooltip = [alt.Tooltip("titolo:N", title="Provvedimento"), alt.Tooltip("data:N", title="Data"),
               alt.Tooltip("tipologia:N", title="Tipologia"), alt.Tooltip("argomenti_testo:N", title="Argomenti"),
               alt.Tooltip("simili:Q", title="Quasi identici")]
    choose = alt.selection_point(name="doc", fields=["i", "vicini"], on="click", nearest=True, empty=False)
    return alt.Chart(data).mark_point(filled=True).encode(
        x=alt.X("x:Q", axis=None, scale=alt.Scale(zero=False, nice=False)),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(zero=False, nice=False)),
        color=color, size=value(base, 260, 110), opacity=value(0.75, 1, 1, 0.22),
        stroke=value("transparent", C["ink"], C["ink"]), strokeWidth=value(0, 2, 1.5),
        shape=value("circle", "diamond", "circle"), order=order, tooltip=tooltip,
    ).add_params(choose).properties(height=640).interactive(name="zoom")  # a fixed name: the same chart every run


def doc_panel(df: pd.DataFrame, i: int, nn_idx: np.ndarray, nn_cos: np.ndarray) -> None:
    d = df.iloc[i]
    tags = " ".join(f"#{html.escape(a)}" for a in d["argomenti"][:5])
    twins = f"<br><small>+{num(d['simili'])} provvedimenti quasi identici, rappresentati da questo</small>" \
        if d["simili"] else ""
    html_md(f"<div class='doc'><p><a href='{html.escape(d['url'])}' target='_blank'><b>{html.escape(d['titolo'])}</b>"
            f"</a><br><small>{html.escape(d['data'])} · {html.escape(d['tipologia'])}</small>"
            + (f"<br><small>{tags}</small>" if tags else "") + f"{twins}</p></div>")
    html_md("<p><b>I 10 più vicini</b> · coseno nello spazio completo dell'embedding, non sulla mappa</p>")
    items = []
    for j, cos in zip(nn_idx[i], nn_cos[i]):
        n = df.iloc[int(j)]
        items.append(f"<li><a href='{html.escape(n['url'])}' target='_blank'>{html.escape(n['titolo'])}</a>"
                     f"<br><small>{html.escape(n['data'])} · {html.escape(n['gruppo'])} · coseno {dec(cos, 3)}</small></li>")
    html_md("<ol class='nn'>" + "".join(items) + "</ol>")


def comparison(index: dict, scope: str | None, shown: str) -> None:
    rows = []
    for snapshot, label in e.SNAPSHOT_LABELS.items():
        for key, s in index.items():
            if s["snapshot"] != snapshot or s["scope"] != scope:
                continue
            rows.append({"Versione": label, "Modello": b.MODELS[s["model"]]["label"],
                         "Provvedimenti": num(s["documents"]), "Passaggi": num(s["passages"]),
                         "Varianza in 2D (PCA)": f"{dec(100 * s['var2d'], 0)}%",
                         "AUC citazioni": dec(s["auc"]) if "auc" in s else "—",
                         "Recall@10 citazioni": f"{dec(100 * s['recall10'], 0)}%" if "recall10" in s else "—",
                         "Collegamenti": num(s.get("links", 0)), "_key": key})
    df = pd.DataFrame(rows)
    st.dataframe(df.drop(columns="_key").style.apply(
        lambda r: ["font-weight: 600" if df.at[r.name, "_key"] == shown else "" for _ in r], axis=1),
        hide_index=True, width="stretch", height=36 * (len(df) + 1) + 3)
    st.caption("AUC: probabilità che due provvedimenti collegati da una citazione siano più simili di due presi a caso "
               "(0,5 = nessun segnale). Recall@10: quota delle citazioni il cui provvedimento citato è tra i 10 più "
               "vicini di quello che cita. Misurate nello spazio completo; esclusi i rinvii ai regolamenti interni del "
               "Garante. Varianza in 2D: quanto della dispersione dei vettori tengono le due componenti della PCA. "
               "Il confronto vale tra versioni con gli stessi provvedimenti (v0–v1, v2–v5, v6–attuale): tra più "
               "provvedimenti è più difficile che quello citato sia tra i 10 più vicini.")


# ----------------------------------------------------------------------------- page
index = read_index()
st.title("Mappe degli embedding")
if not index:
    st.error("Mappe non calcolate: eseguire `python phase_maps.py`.")
    st.stop()
html_md("<p class='lead'>Ogni punto è un provvedimento, collocato dal suo embedding (la media dei vettori dei suoi "
        "passaggi): vicini sulla mappa vuol dire simili per il modello. Le mappe di tutte le versioni sono orientate "
        "come quella attuale, così uno stesso provvedimento resta più o meno nella stessa zona.</p>")

versions = [s for s in e.SNAPSHOT_LABELS if any(m["snapshot"] == s for m in index.values())]
if "_from_url" not in st.session_state:  # first run: the choices in the address, so a map can be bookmarked
    st.session_state["_from_url"] = True
    q = st.query_params
    st.session_state["versione"] = {short(s): s for s in versions}.get(q.get("versione"), e.NOW)
    for k in ("modello", "proiezione", "colori"):
        if k in q:
            st.session_state[k] = q[k]  # pick() falls back to the default when the value isn't offered
    if q.get("passaggi") == "violazioni":
        st.session_state["ambito"] = "violation"
version = st.select_slider("Versione dell'indice", versions, format_func=short, key="versione")

models = [m for m in b.MODELS if pm.map_key(version, m, None) in index]
scopes = [s for s in SCOPE_LABELS if models and pm.map_key(version, models[0], s) in index]
default_model = pm.REFERENCE[1] if pm.REFERENCE[1] in models else models[0]
with st.container(horizontal=True, gap="large"):
    model = pick("Modello", models, default_model, "modello", format_func=lambda m: b.MODELS[m]["label"])
    scope = pick("Passaggi", scopes, None, "ambito", format_func=SCOPE_LABELS.get)
    proj = pick("Proiezione", list(PROJECTIONS), "tsne", "proiezione", format_func=PROJECTIONS.get)
    mode = pick("Colori", list(COLOR_MODES), "Tipologia", "colori")
st.query_params.from_dict({k: v for k, v in {
    "versione": short(version), "modello": model if model != default_model else None,
    "passaggi": "violazioni" if scope else None, "proiezione": proj if proj != "tsne" else None,
    "colori": mode if mode != "Tipologia" else None}.items() if v})

key = pm.map_key(version, model, scope)
stats = index[key]
df, nn_idx, nn_cos = load_map(key, version)
topic = None
if mode == "Argomento":
    counts = pd.Series([a for tags in df["argomenti"] for a in tags]).value_counts()
    topics = list(counts.index)
    topic = st.selectbox("Argomento in evidenza", topics, index=topics.index(DEFAULT_TOPIC) if DEFAULT_TOPIC in topics
                         else 0, format_func=lambda t: f"{t} ({num(int(counts[t]))})", key="argomento")

st.subheader(e.SNAPSHOT_LABELS[version])
facts = [f"{num(stats['documents'])} provvedimenti", f"{num(stats['passages'])} passaggi",
         b.MODELS[model]["label"], SCOPE_LABELS[scope].lower()]
if proj == "pca":
    facts.append(f"le due componenti tengono il {dec(100 * stats['var2d'], 0)}% della varianza")
st.caption(" · ".join(facts))

chart_box, side = st.columns([2.3, 1], gap="large")
with chart_box:
    event = st.altair_chart(map_chart(df, proj, mode, topic, nn_idx), width="stretch", key=f"map_{key}_{proj}",
                            on_select="rerun", selection_mode="doc")
    points = event.selection.get("doc", [])
    chosen = int(points[0]["i"]) if points and 0 <= int(points[0].get("i", -1)) < len(df) else None
    st.caption("Passa sopra un punto per i dettagli · clic per i 10 provvedimenti più vicini · rotella per lo zoom · "
               "trascina per spostarti · doppio clic per deselezionare e tornare alla vista intera")
with side:
    if chosen is not None:
        doc_panel(df, chosen, nn_idx, nn_cos)
    else:
        if NOTES.get(version):
            st.info(NOTES[version], icon=":material/visibility:")
        if proj == "pca":
            st.caption("PCA è una proiezione lineare: conserva le grandi distanze (un gruppo di provvedimenti quasi "
                       "identici si stacca dal resto) ma due componenti tengono solo una parte della varianza.")
        else:
            st.caption("t-SNE conserva i vicinati: i provvedimenti sullo stesso tema si raccolgono, ma le distanze "
                       "tra gruppi lontani e la loro dimensione non vanno lette.")

st.subheader("Da una versione all'altra")
comparison(index, scope, key)
