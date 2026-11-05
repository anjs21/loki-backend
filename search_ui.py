"""
search_ui.py - pieces shared by the pages of the Streamlit app (app.py, pages/): the embedding models, loaded once
for the life of the server whichever page asks first, and the display helpers for passages.
"""
from __future__ import annotations

import html
import re

import streamlit as st

import build_index as b

STOPWORDS = {"della", "delle", "dello", "degli", "nella", "nelle", "sulla", "sulle", "dalla", "dalle", "come",
             "anche", "sono", "questo", "questa", "quale", "quali", "with", "from", "that", "their", "after"}

STYLE = """
<style>
.passage { font-size: 0.92rem; line-height: 1.5; padding: 0.5rem 0.75rem; border-left: 3px solid #9aa5b1;
           background: rgba(127,127,127,0.06); border-radius: 4px; margin-bottom: 0.5rem; }
mark { background: #ffe58f; color: inherit; padding: 0 1px; border-radius: 2px; }
.tag { display: inline-block; font-size: 0.78rem; padding: 1px 8px; margin: 0 4px 4px 0; border-radius: 10px;
       background: rgba(100,130,180,0.15); }
</style>
"""


@st.cache_resource(show_spinner="Caricamento del modello di embedding...")
def get_model(key: str):
    model = b.load_model(key)
    b.encode_queries(model, key, ["riscaldamento"])  # the first encode pays one-off CUDA setup (~0.25 s)
    return model


def md_escape(text: str) -> str:
    """Escape Markdown in link text."""
    return re.sub(r"([\[\]*_`$])", r"\\\1", text)


def query_words(query: str) -> list[str]:
    """Words of 4+ letters worth marking, longest first."""
    words = {w.lower() for w in re.findall(r"\w{4,}", query) if w.lower() not in STOPWORDS}
    return sorted(words, key=len, reverse=True)


def highlight(text: str, query: str) -> str:
    """HTML-escaped text with query words marked (words of 4+ letters, prefix match to catch inflections)."""
    out = html.escape(text).replace("\n", "<br>")
    for w in query_words(query):
        stem = re.escape(w[:max(4, len(w) - 2)])
        out = re.sub(rf"(?i)\b({stem}\w*)", r"<mark>\1</mark>", out)
    return out


_TAG = re.compile(r"(<[^>]*>)")


def spans(markup: str) -> str:
    """The HTML with every run of text that shares its element with tags put in a <span> of its own.

    Browser page translation (Chrome's "Translate this page") swaps each text node for <font> elements of its own.
    When React later updates an element holding both text and tags (a passage with <mark> and <br>), it looks for the
    text nodes it made, which are gone, and the element fails with "NotFoundError: Failed to execute 'removeChild' on
    'Node'". With no element holding both, React only adds and removes whole elements, which the translator leaves in
    place, and sets the text of each as a whole."""
    parts = _TAG.split(markup)
    out = []
    for i, p in enumerate(parts):
        if not p or p.startswith("<"):
            out.append(p)
            continue
        prev, nxt = parts[i - 1] if i else "", parts[i + 1] if i + 1 < len(parts) else ""
        alone = (prev.startswith("<") and not prev.startswith(("</", "<br")) and not prev.endswith("/>")
                 and nxt.startswith("</"))
        out.append(p if alone else f"<span>{p}</span>")
    return "".join(out)


def html_md(markup: str, where=None) -> None:
    """Render HTML so that browser page translation can't break its later updates (see spans). It must open with a
    block tag (<p>, <div>, <ul>) and have no blank line, so Markdown leaves it as it is."""
    (where or st).markdown(spans(markup), unsafe_allow_html=True)


def num(x) -> str:
    """Italian number format: 74.633, 4,76."""
    if isinstance(x, float) and not x.is_integer():
        return f"{x:,.2f}".replace(",", " ").replace(".", ",").replace(" ", ".")
    return f"{int(x):,}".replace(",", ".")


def dec(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def chart_colors() -> dict[str, str]:
    """Chart colours from the reference palette, stepped for the viewer's light or dark theme: the first three
    categorical slots (blue, orange, aqua: the most a scatter can tell apart), de-emphasis, ink."""
    dark = getattr(getattr(st.context, "theme", None), "type", None) == "dark"
    return {"accent": "#3987e5" if dark else "#2a78d6", "second": "#d95926" if dark else "#eb6834",
            "third": "#199e70" if dark else "#1baf7a", "muted_mark": "#52514e" if dark else "#c3c2b7",
            "ink": "#f0efec" if dark else "#0b0b0b", "ink_muted": "#898781", "dark": dark}


IT_AXIS = "replace(datum.label, '.', ',')"  # Italian decimal comma on chart axes


def inline_md(text: str) -> str:
    """HTML for the inline Markdown of the page texts: `code`, **bold**, *italics*."""
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    return re.sub(r"(?<![\w*])\*([^*\s][^*]*)\*(?![\w*])", r"<i>\1</i>", out)
