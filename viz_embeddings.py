#!/usr/bin/env python3
"""
viz_embeddings.py - interactive 2D map of the POC index (PCA of document embeddings) with the citation graph,
plus a check of whether decisions that cite each other are close in the full embedding space.

  python viz_embeddings.py                        # -> data/poc/embedding_map.html (open in a browser)
  python viz_embeddings.py --sections violation   # -> data/poc/embedding_map_violation.html

Document vector = mean of its chunk vectors, re-normalised. With --sections, only the chunks of those sections
(violation: motivazione + dispositivo of case decisions, so no opinions or general acts; activity: sintesi +
fatti) count, and decisions without any are left out.

Citation links come from data/docs.jsonl (`cites`, `see_also`, and "doc. web n. ..." references in the text), mapped
onto the indexed documents (a link to a near-duplicate is a link to its template representative).

The 2D picture is only a projection: two principal components keep a small share of the variance, so pairs that
look far apart can still be close. The statistics (AUC, recall@10, median rank) are computed in the full space.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

import build_index as b

SCOPE_TEXT = {None: ("", "mean of its chunk embeddings"),
              "violation": (" — violation sections only",
                            "mean of the embeddings of its violation chunks only: motivazione and dispositivo; "
                            "only decisions on a case, without opinions, deliberations and general acts"),
              "activity": (" — fact sections only",
                           "mean of the embeddings of its fact chunks only: sintesi and fatti; "
                           "decisions without such chunks are left out")}
# The Garante's own procedural rules (e.g. Regolamento n. 1/2019, cited by ~550 decisions for "art. 17 ... annotazione
# nel registro interno") say nothing about what a case is about, so links to them are not "should be close" pairs.
PROCEDURAL_TYPE = "Regolamento del Garante"
N_RANDOM_PAIRS = 20000
HIST_BINS = 40


def load_docs(scope: str | None = None):
    """Chunks in scope, their rows in the embedding files, and the documents they belong to."""
    chunks = []
    for line in (b.OUT / "chunks.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        c.pop("embed_text", None)
        chunks.append(c)
    rows = np.flatnonzero(b.scope_mask(chunks, scope))
    chunks = [chunks[i] for i in rows]
    doc_ids = list(dict.fromkeys(c["docweb_id"] for c in chunks))
    pos = {d: i for i, d in enumerate(doc_ids)}
    chunk_doc = np.array([pos[c["docweb_id"]] for c in chunks])
    first = {}
    for c in chunks:
        first.setdefault(c["docweb_id"], c)
    nodes = [{"id": d, "t": first[d]["title"], "d": first[d]["date"], "ty": ", ".join(first[d]["tipologia"]),
              "a": ", ".join(first[d]["argomenti"][:3]), "n": first[d].get("similar_count", 0)} for d in doc_ids]
    return chunks, rows, chunk_doc, doc_ids, pos, nodes


def citation_edges(doc_ids: list[int], pos: dict[int, int]) -> list[tuple[int, int]]:
    rep_of = {d: d for d in doc_ids}
    dup_path = b.OUT / "duplicates.json"
    if dup_path.exists():
        for rep, twins in json.loads(dup_path.read_text(encoding="utf-8")).items():
            if int(rep) in pos:  # with --sections the representative may be left out
                for t in twins:
                    rep_of[t["docweb_id"]] = int(rep)
    wanted = set(doc_ids)
    edges = set()
    for line in (b.DATA / "docs.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["docweb_id"] not in wanted:
            continue
        for target in r["cites"] + r["see_also"] + r["refs_in_text"]:
            t = rep_of.get(target)
            if t is not None and t != r["docweb_id"]:
                i, j = pos[r["docweb_id"]], pos[t]
                edges.add((min(i, j), max(i, j)))
    return sorted(edges)


def doc_vectors(emb: np.ndarray, chunk_doc: np.ndarray, n_docs: int) -> np.ndarray:
    v = np.zeros((n_docs, emb.shape[1]), dtype=np.float32)
    np.add.at(v, chunk_doc, emb.astype(np.float32))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """P(score of a linked pair > score of a random pair), ties counted half."""
    neg = np.sort(neg_scores)
    lo = np.searchsorted(neg, pos_scores, side="left")
    hi = np.searchsorted(neg, pos_scores, side="right")
    return float(((lo + hi) / 2).mean() / len(neg))


def model_view(key: str, rows, chunk_doc, n_docs: int, edges: list[tuple[int, int]], rng) -> dict:
    v = doc_vectors(np.load(b.embeddings_path(key))[rows], chunk_doc, n_docs)
    e = np.array(edges)
    linked_sim = (v[e[:, 0]] * v[e[:, 1]]).sum(1)
    ri = rng.integers(0, n_docs, (N_RANDOM_PAIRS, 2))
    ri = ri[ri[:, 0] != ri[:, 1]]
    random_sim = (v[ri[:, 0]] * v[ri[:, 1]]).sum(1)

    # Citation retrieval: using a decision's vector as the query, where do the decisions it is linked to rank?
    ranks = []
    sims_all = v @ v.T
    np.fill_diagonal(sims_all, -np.inf)
    for a, c in np.concatenate([e, e[:, ::-1]]):
        ranks.append(int((sims_all[a] > sims_all[a, c]).sum()) + 1)
    ranks = np.array(ranks)
    n_e = len(e)  # ranks[:n_e]: target j seen from i; ranks[n_e:]: i seen from j

    # PCA to 2D
    centred = v - v.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    xy = centred @ vt[:2].T
    var = (s ** 2) / (s ** 2).sum()
    d2_linked = np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1)
    d2_random = np.linalg.norm(xy[ri[:, 0]] - xy[ri[:, 1]], axis=1)

    lo, hi = float(min(linked_sim.min(), random_sim.min())), float(max(linked_sim.max(), random_sim.max()))
    bins = np.linspace(lo, hi, HIST_BINS + 1)
    h_linked, _ = np.histogram(linked_sim, bins, density=True)
    h_random, _ = np.histogram(random_sim, bins, density=True)
    return {
        "label": b.MODELS[key]["label"],
        "xy": np.round(xy, 4).tolist(),
        "stats": {
            "var2d": float(var[:2].sum()), "var_pc1": float(var[0]), "var_pc2": float(var[1]),
            "auc": auc(linked_sim, random_sim), "auc2d": auc(-d2_linked, -d2_random),
            "recall10": float((ranks <= 10).mean()), "median_rank": float(np.median(ranks)),
            "median_linked": float(np.median(linked_sim)), "median_random": float(np.median(random_sim)),
        },
        # per link, full-space cosine and the rank of each end seen from the other, for the side panel
        "edge_cos": np.round(linked_sim, 3).tolist(), "edge_rank_ij": ranks[:n_e].tolist(),
        "edge_rank_ji": ranks[n_e:].tolist(),
        "hist": {"x": np.round((bins[:-1] + bins[1:]) / 2, 4).tolist(),
                 "linked": np.round(h_linked, 4).tolist(), "random": np.round(h_random, 4).tolist()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", choices=b.SCOPES, default=None, help="map only these sections of each decision")
    args = ap.parse_args()
    out_html = b.OUT / (f"embedding_map_{args.sections}.html" if args.sections else "embedding_map.html")
    chunks, rows, chunk_doc, doc_ids, pos, nodes = load_docs(args.sections)
    all_edges = citation_edges(doc_ids, pos)
    procedural = {i for i, n in enumerate(nodes) if PROCEDURAL_TYPE.lower() in n["ty"].lower()}
    edges = [e for e in all_edges if e[0] not in procedural and e[1] not in procedural]
    rng = np.random.default_rng(0)
    models = {k: model_view(k, rows, chunk_doc, len(doc_ids), edges, rng)
              for k in b.MODELS if b.embeddings_path(k).exists()}
    payload = {"nodes": nodes, "edges": edges, "models": models, "default": b.DEFAULT_MODEL,
               "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
               "n_chunks": len(chunks), "excluded_links": len(all_edges) - len(edges),
               "excluded_titles": [nodes[i]["t"] for i in sorted(procedural)]}
    title_suffix, vector_text = SCOPE_TEXT[args.sections]
    html = (TEMPLATE.replace("__TITLE_SUFFIX__", title_suffix).replace("__VECTOR_TEXT__", vector_text)
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")))
    out_html.write_text(html, encoding="utf-8")
    print(f"{len(nodes)} documents ({len(chunks)} chunks), {len(edges)} citation links ({len(all_edges) - len(edges)} "
          f"links to {len(procedural)} procedural regulations excluded) -> {out_html}")
    for k, m in models.items():
        s = m["stats"]
        print(f"  {k:7s} PCA var(2D) {s['var2d']:.1%} | AUC linked>random {s['auc']:.3f} (2D: {s['auc2d']:.3f}) | "
              f"recall@10 {s['recall10']:.1%} | median rank {s['median_rank']:.0f} | "
              f"median cos linked {s['median_linked']:.3f} vs random {s['median_random']:.3f}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Embedding Map__TITLE_SUFFIX__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #eb6834; --neutral: #c3c2b7;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #d95926; --neutral: #4a4a47;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #d95926; --neutral: #4a4a47;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1280px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--ink-2); margin: 0 0 16px; max-width: 900px; }
.controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 16px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { border: 0; background: transparent; color: var(--ink-2); padding: 6px 12px; font: inherit; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--surface-1); color: var(--ink); font-weight: 600;
                                   box-shadow: inset 0 0 0 1px var(--border); }
label.chk { display: inline-flex; gap: 6px; align-items: center; color: var(--ink-2); cursor: pointer; }
input[type=search] { font: inherit; padding: 6px 10px; border: 1px solid var(--border); border-radius: 8px;
                     background: var(--surface-1); color: var(--ink); min-width: 260px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 16px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.tile .lab { color: var(--ink-2); font-size: 12px; }
.tile .val { font-size: 24px; font-weight: 600; margin-top: 2px; }
.tile .note { color: var(--muted); font-size: 12px; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 16px; }
@media (max-width: 900px) { .layout { grid-template-columns: 1fr; } }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px; }
.card h2 { font-size: 14px; font-weight: 600; margin: 0 0 8px; }
.legend { display: flex; flex-wrap: wrap; gap: 14px; color: var(--ink-2); font-size: 12px; margin: 0 0 8px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }
.legend i.line { width: 16px; height: 2px; border-radius: 1px; vertical-align: 3px; }
#map { width: 100%; height: auto; aspect-ratio: 4 / 3; display: block; touch-action: none; cursor: grab; }
#map.dragging { cursor: grabbing; }
#panel { font-size: 13px; color: var(--ink-2); }
#panel a { color: var(--ink); }
#panel ul { padding-left: 18px; margin: 6px 0; max-height: 360px; overflow: auto; }
#panel li { margin-bottom: 4px; }
.hint { color: var(--muted); font-size: 12px; margin-top: 6px; }
#tip { position: fixed; pointer-events: none; background: var(--surface-1); color: var(--ink);
       border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 12px; max-width: 340px;
       box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; z-index: 10; }
#tip .m { color: var(--ink-2); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); }
th { color: var(--ink-2); font-weight: 600; }
td.num { font-variant-numeric: tabular-nums; }
.below { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; margin-top: 16px; }
@media (max-width: 900px) { .below { grid-template-columns: 1fr; } }
#hist { width: 100%; height: auto; aspect-ratio: 16 / 9; display: block; }
</style>
</head>
<body>
<main>
  <h1>Embedding map of the Garante decisions__TITLE_SUFFIX__</h1>
  <p class="sub">Each dot is one decision (__VECTOR_TEXT__), projected to two dimensions with PCA.
    Lines are citations between decisions. Two components keep only a small share of the variance, so judge
    "do cited decisions sit close?" from the tiles, which are measured in the full embedding space.
    <span id="excl"></span></p>

  <div class="controls">
    <div class="seg" id="modelSeg" role="group" aria-label="Embedding model"></div>
    <label class="chk"><input type="checkbox" id="showEdges"> Show all citation links</label>
    <input type="search" id="find" placeholder="Find by doc web id or title…" aria-label="Find a decision">
  </div>

  <div class="tiles" id="tiles"></div>

  <div class="layout">
    <div class="card">
      <div class="legend">
        <span><i style="background:var(--series-1)"></i>Has citation links</span>
        <span><i style="background:var(--neutral)"></i>No links in the index</span>
        <span><i style="background:var(--series-2)"></i>Linked to the selected decision</span>
        <span><i class="line" style="background:var(--series-2)"></i>Citation of the selected decision</span>
      </div>
      <svg id="map" role="img" aria-label="PCA scatter plot of decision embeddings"></svg>
      <div class="hint">Hover for details · click a dot to show its citations · scroll to zoom · drag to pan · double-click to reset</div>
    </div>
    <div class="card" id="panel"><h2>Selected decision</h2><p>Click a dot to see the decisions it cites or is cited by,
      and how close they are in the full embedding space.</p></div>
  </div>

  <div class="below">
    <div class="card">
      <h2>Cosine similarity: linked pairs vs random pairs</h2>
      <div class="legend">
        <span><i class="line" style="background:var(--series-1)"></i>Decisions linked by a citation</span>
        <span><i class="line" style="background:var(--neutral)"></i>Random pairs (baseline)</span>
      </div>
      <svg id="hist" role="img" aria-label="Density of cosine similarity for linked and random pairs"></svg>
    </div>
    <div class="card">
      <h2>Model comparison (table view)</h2>
      <table id="cmp"></table>
      <p class="hint">AUC: chance a linked pair is more similar than a random pair (0.5 = no signal).
        Recall@10: share of citations whose target is among the 10 nearest decisions of the citing one.</p>
    </div>
  </div>
</main>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const nodes = DATA.nodes, edges = DATA.edges;
const adj = nodes.map(() => []), adjEdge = nodes.map(() => []);
edges.forEach(([i, j], k) => { adj[i].push(j); adj[j].push(i); adjEdge[i].push(k); adjEdge[j].push(k); });
let model = DATA.models[DATA.default] ? DATA.default : Object.keys(DATA.models)[0];
let selected = null, view = null;
const W = 800, H = 600, PAD = 24;
const svg = document.getElementById("map");
svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const url = id => `https://www.garanteprivacy.it/home/docweb/-/docweb-display/docweb/${id}`;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const pct = x => (100 * x).toFixed(1) + "%";

// --- model toggle + tiles + comparison table
const seg = document.getElementById("modelSeg");
Object.entries(DATA.models).forEach(([k, m]) => {
  const btn = document.createElement("button");
  btn.textContent = m.label; btn.dataset.k = k;
  btn.onclick = () => { model = k; render(); };
  seg.appendChild(btn);
});
function tiles() {
  const s = DATA.models[model].stats;
  const items = [
    ["Variance kept by 2D PCA", pct(s.var2d), `PC1 ${pct(s.var_pc1)} · PC2 ${pct(s.var_pc2)}`],
    ["Linked closer than random (AUC)", s.auc.toFixed(3), `in 2D only: ${s.auc2d.toFixed(3)}`],
    ["Citation recall@10", pct(s.recall10), `${edges.length.toLocaleString()} links, both directions`],
    ["Median rank of cited decision", Math.round(s.median_rank).toLocaleString(), `out of ${(nodes.length - 1).toLocaleString()}`],
  ];
  document.getElementById("tiles").innerHTML = items.map(([l, v, n]) =>
    `<div class="tile"><div class="lab">${l}</div><div class="val">${v}</div><div class="note">${n}</div></div>`).join("");
  const rows = Object.values(DATA.models).map(m => `<tr><td>${m.label}</td><td class="num">${pct(m.stats.var2d)}</td>
    <td class="num">${m.stats.auc.toFixed(3)}</td><td class="num">${pct(m.stats.recall10)}</td>
    <td class="num">${Math.round(m.stats.median_rank)}</td><td class="num">${m.stats.median_linked.toFixed(3)} / ${m.stats.median_random.toFixed(3)}</td></tr>`).join("");
  document.getElementById("cmp").innerHTML = `<thead><tr><th>Model</th><th>2D var.</th><th>AUC</th><th>Recall@10</th>
    <th>Median rank</th><th>Median cos linked / random</th></tr></thead><tbody>${rows}</tbody>`;
  seg.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.k === model));
}

// --- scatter
const root = el("g", {}, svg), gEdges = el("g", {}, root), gSel = el("g", {}, root), gDots = el("g", {}, root);
let sx, sy, dots = [];
function scales() {
  const xy = DATA.models[model].xy;
  const xs = xy.map(p => p[0]), ys = xy.map(p => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  sx = x => PAD + (x - x0) / (x1 - x0) * (W - 2 * PAD);
  sy = y => H - PAD - (y - y0) / (y1 - y0) * (H - 2 * PAD);
}
function drawDots() {
  gDots.innerHTML = ""; dots = [];
  const xy = DATA.models[model].xy;
  // unlinked first so the linked ones sit on top
  const order = nodes.map((_, i) => i).sort((a, b) => (adj[a].length > 0) - (adj[b].length > 0));
  order.forEach(i => {
    const c = el("circle", {cx: sx(xy[i][0]), cy: sy(xy[i][1]), r: 4, "stroke-width": 1.5,
      fill: adj[i].length ? "var(--series-1)" : "var(--neutral)", stroke: "var(--surface-1)"}, gDots);
    c.dataset.i = i; dots[i] = c;
  });
}
function drawAllEdges() {
  gEdges.innerHTML = "";
  if (!document.getElementById("showEdges").checked) return;
  const xy = DATA.models[model].xy;
  edges.forEach(([i, j]) => el("line", {x1: sx(xy[i][0]), y1: sy(xy[i][1]), x2: sx(xy[j][0]), y2: sy(xy[j][1]),
    stroke: "var(--series-1)", "stroke-opacity": 0.12, "stroke-width": 1, "vector-effect": "non-scaling-stroke"}, gEdges));
}
function drawSelection() {
  gSel.innerHTML = "";
  dots.forEach((c, i) => { c.setAttribute("fill", adj[i].length ? "var(--series-1)" : "var(--neutral)");
                           c.setAttribute("r", 4); c.setAttribute("stroke", "var(--surface-1)"); });
  if (selected === null) return;
  const xy = DATA.models[model].xy, s = selected;
  adj[s].forEach(j => {
    el("line", {x1: sx(xy[s][0]), y1: sy(xy[s][1]), x2: sx(xy[j][0]), y2: sy(xy[j][1]), stroke: "var(--series-2)",
      "stroke-width": 2, "stroke-linecap": "round", "vector-effect": "non-scaling-stroke"}, gSel);
    dots[j].setAttribute("fill", "var(--series-2)"); dots[j].setAttribute("r", 5); gDots.appendChild(dots[j]);
  });
  dots[s].setAttribute("r", 7); dots[s].setAttribute("fill", adj[s].length ? "var(--series-1)" : "var(--neutral)");
  dots[s].setAttribute("stroke", "var(--ink)"); gDots.appendChild(dots[s]);
}
function panel() {
  const p = document.getElementById("panel");
  if (selected === null) return;
  const n = nodes[selected], m = DATA.models[model];
  // full-space closeness of each linked decision: cosine, and its rank among the selected one's neighbours
  const links = adjEdge[selected].map(k => { const [i, j] = edges[k], other = i === selected ? j : i;
    return {j: other, cos: m.edge_cos[k], rank: i === selected ? m.edge_rank_ij[k] : m.edge_rank_ji[k]}; })
    .sort((a, b) => a.rank - b.rank);
  const items = links.map(({j, cos, rank}) => `<li>${esc(nodes[j].d)} · <a href="${url(nodes[j].id)}" target="_blank" rel="noopener">${esc(nodes[j].t)}</a>
      <span class="m"> — cos ${cos.toFixed(3)} · #${rank} nearest</span></li>`).join("");
  p.innerHTML = `<h2>Selected decision</h2>
    <p><a href="${url(n.id)}" target="_blank" rel="noopener"><b>${esc(n.t)}</b></a><br>${esc(n.d)} · ${esc(n.ty)}
      ${n.a ? "<br>#" + esc(n.a).replaceAll(", ", " #") : ""}${n.n ? `<br>+${n.n} near-identical decisions (template)` : ""}</p>
    <p>${adj[selected].length ? `${adj[selected].length} linked decision${adj[selected].length > 1 ? "s" : ""} (orange), nearest first.
      "#5 nearest" = 5th most similar of all ${(nodes.length - 1).toLocaleString()} decisions, in the full embedding space:` : "No citation links inside the index."}</p>
    <ul>${items}</ul>`;
}

// tooltip, click, zoom/pan
const tip = document.getElementById("tip");
svg.addEventListener("mousemove", e => {
  const c = e.target.closest("circle");
  if (!c) { tip.style.display = "none"; return; }
  const n = nodes[+c.dataset.i];
  tip.innerHTML = `<b>${esc(n.t)}</b><br><span class="m">${esc(n.d)} · ${esc(n.ty)}</span>
    ${n.a ? `<br><span class="m">#${esc(n.a).replaceAll(", ", " #")}</span>` : ""}
    <br><span class="m">${adj[+c.dataset.i].length} citation link${adj[+c.dataset.i].length === 1 ? "" : "s"}</span>`;
  tip.style.display = "block";
  const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(e.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
});
svg.addEventListener("mouseleave", () => tip.style.display = "none");
let drag = null, moved = false;
view = {k: 1, x: 0, y: 0};
const applyView = () => root.setAttribute("transform", `translate(${view.x} ${view.y}) scale(${view.k})`);
const toSvg = e => { const r = svg.getBoundingClientRect(); return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height]; };
svg.addEventListener("wheel", e => {
  e.preventDefault();
  const [px, py] = toSvg(e), f = e.deltaY < 0 ? 1.25 : 0.8, k = Math.min(40, Math.max(1, view.k * f));
  view.x = px - (px - view.x) * k / view.k; view.y = py - (py - view.y) * k / view.k; view.k = k; applyView();
}, {passive: false});
svg.addEventListener("pointerdown", e => { drag = toSvg(e); moved = false; svg.classList.add("dragging"); });
window.addEventListener("pointermove", e => {
  if (!drag) return; const [px, py] = toSvg(e);
  if (Math.abs(px - drag[0]) + Math.abs(py - drag[1]) > 2) moved = true;
  view.x += px - drag[0]; view.y += py - drag[1]; drag = [px, py]; applyView();
});
window.addEventListener("pointerup", e => {
  svg.classList.remove("dragging");
  if (drag && !moved) { const c = e.target.closest && e.target.closest("circle");
    selected = c ? +c.dataset.i : null; drawSelection(); panel(); }
  drag = null;
});
svg.addEventListener("dblclick", () => { view = {k: 1, x: 0, y: 0}; applyView(); });
document.getElementById("showEdges").addEventListener("change", drawAllEdges);
document.getElementById("find").addEventListener("change", e => {
  const q = e.target.value.trim().toLowerCase(); if (!q) return;
  const i = nodes.findIndex(n => String(n.id) === q || n.t.toLowerCase().includes(q));
  if (i >= 0) { selected = i; drawSelection(); panel(); }
});

// --- histogram (density lines)
function hist() {
  const h = DATA.models[model].hist, hs = document.getElementById("hist"), HW = 640, HH = 360,
        L = 44, R = 12, T = 12, B = 34;
  hs.setAttribute("viewBox", `0 0 ${HW} ${HH}`); hs.innerHTML = "";
  const x0 = h.x[0], x1 = h.x[h.x.length - 1], ymax = Math.max(...h.linked, ...h.random) * 1.08;
  const X = x => L + (x - x0) / (x1 - x0) * (HW - L - R), Y = y => HH - B - y / ymax * (HH - T - B);
  for (let k = 0; k <= 4; k++) { const yv = ymax * k / 4;
    el("line", {x1: L, x2: HW - R, y1: Y(yv), y2: Y(yv), stroke: "var(--grid)", "stroke-width": 1}, hs);
    el("text", {x: L - 6, y: Y(yv) + 4, "text-anchor": "end", "font-size": 11, fill: "var(--muted)"}, hs).textContent = yv.toFixed(1); }
  for (let k = 0; k <= 5; k++) { const xv = x0 + (x1 - x0) * k / 5;
    el("text", {x: X(xv), y: HH - B + 16, "text-anchor": "middle", "font-size": 11, fill: "var(--muted)"}, hs).textContent = xv.toFixed(2); }
  el("text", {x: (L + HW - R) / 2, y: HH - 4, "text-anchor": "middle", "font-size": 11, fill: "var(--ink-2)"}, hs).textContent = "cosine similarity between the two decisions";
  el("line", {x1: L, x2: HW - R, y1: Y(0), y2: Y(0), stroke: "var(--axis)", "stroke-width": 1}, hs);
  const path = ys => ys.map((y, k) => `${k ? "L" : "M"}${X(h.x[k]).toFixed(1)},${Y(y).toFixed(1)}`).join("");
  el("path", {d: path(h.random), fill: "none", stroke: "var(--neutral)", "stroke-width": 2, "stroke-linejoin": "round"}, hs);
  el("path", {d: path(h.linked), fill: "none", stroke: "var(--series-1)", "stroke-width": 2, "stroke-linejoin": "round"}, hs);
  const s = DATA.models[model].stats;
  [[s.median_random, "var(--neutral)", "random median"], [s.median_linked, "var(--series-1)", "linked median"]].forEach(([v, col, lab], k) => {
    el("line", {x1: X(v), x2: X(v), y1: T, y2: HH - B, stroke: col, "stroke-width": 1}, hs);
    el("text", {x: X(v) + 4, y: T + 12 + k * 14, "font-size": 11, fill: "var(--ink-2)"}, hs).textContent = `${lab} ${v.toFixed(3)}`;
  });
  // hover: value readout per bin
  const ro = el("text", {x: HW - R, y: T + 12, "text-anchor": "end", "font-size": 11, fill: "var(--ink-2)"}, hs);
  const cross = el("line", {y1: T, y2: HH - B, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden"}, hs);
  hs.onmousemove = e => { const r = hs.getBoundingClientRect(), px = (e.clientX - r.left) * HW / r.width;
    const k = Math.max(0, Math.min(h.x.length - 1, Math.round((px - L) / (HW - L - R) * (h.x.length - 1))));
    cross.setAttribute("x1", X(h.x[k])); cross.setAttribute("x2", X(h.x[k])); cross.setAttribute("visibility", "visible");
    ro.textContent = `cos ${h.x[k].toFixed(3)} · linked ${h.linked[k].toFixed(2)} · random ${h.random[k].toFixed(2)}`; };
  hs.onmouseleave = () => { cross.setAttribute("visibility", "hidden"); ro.textContent = ""; };
}

if (DATA.excluded_links) document.getElementById("excl").textContent =
  `${DATA.excluded_links.toLocaleString()} links to the Garante's own procedural regulations are excluded.`;
function render() { scales(); tiles(); drawDots(); drawAllEdges(); drawSelection(); panel(); hist(); }
// optional URL parameters: ?theme=light|dark, ?model=jasper|bge-m3, ?select=<doc web id>
const params = new URLSearchParams(location.search);
if (["light", "dark"].includes(params.get("theme"))) document.documentElement.dataset.theme = params.get("theme");
if (DATA.models[params.get("model")]) model = params.get("model");
const pre = nodes.findIndex(n => String(n.id) === params.get("select"));
if (pre >= 0) selected = pre;
render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
