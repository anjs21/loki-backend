#!/usr/bin/env python3
"""
phase_maps.py - the embedding maps of every version of the index (the data/poc snapshots, see evolution.py), for the
"Mappe degli embedding" page of the Streamlit app.

  python phase_maps.py            # -> data/poc/maps/ (every version x model x scope; a few minutes)

Each decision is the mean of its passages' vectors, re-normalised, as in viz_embeddings.py; with the violation scope,
of its violation passages only (motivazione and dispositivo of case decisions). Two projections to 2D:
  - PCA, linear: keeps the large distances, so a family of hundreds of near-identical decisions stands apart;
  - t-SNE: keeps the neighbourhoods, so decisions on the same subject gather.
Every map is turned (rotation, reflection and scale only: orthogonal Procrustes) to match the map of the current index
on the decisions both contain, so a decision sits in roughly the same place from one version to the next.

Per map, also: the 10 nearest decisions of each in the full space (the 2D picture is only a projection), and whether
decisions linked by a citation are more similar than random pairs (AUC, recall@10), as in viz_embeddings.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import build_index as b
import evolution as e
import viz_embeddings as viz

MAPS = b.OUT / "maps"
INDEX = MAPS / "index.json"
SCOPES = (None, "violation")
PROJECTIONS = ("pca", "tsne")
REFERENCE = (e.NOW, "bidirlm")      # every map is aligned to this one (same scope and projection)
NEIGHBOURS = 10
PCA_DIMS_FOR_TSNE = 50


def map_key(snapshot: str, model: str, scope: str | None) -> str:
    return f"{snapshot or 'attuale'}__{model}__{scope or 'tutto'}"


def docs_path(snapshot: str) -> Path:
    return MAPS / f"{snapshot or 'attuale'}__docs.json"


# ----------------------------------------------------------------------------- reading (the page)
def read_index() -> dict:
    return json.loads(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else {}


def read_map(key: str) -> dict[str, np.ndarray]:
    with np.load(MAPS / f"{key}.npz") as z:
        return {k: z[k] for k in z.files}


def read_docs(snapshot: str) -> dict[int, dict]:
    return {d["id"]: d for d in json.loads(docs_path(snapshot).read_text(encoding="utf-8"))}


# ----------------------------------------------------------------------------- building
def doc_vectors(snap: e.Snapshot, model: str, scope: str | None) -> tuple[list[int], np.ndarray]:
    """Ids of the decisions with passages in scope and their vectors (mean of those passages, re-normalised)."""
    rows = np.flatnonzero(b.scope_mask(snap.chunks, scope))
    ids = list(dict.fromkeys(snap.chunks[i]["docweb_id"] for i in rows))
    pos = {d: i for i, d in enumerate(ids)}
    chunk_doc = np.array([pos[snap.chunks[i]["docweb_id"]] for i in rows])
    emb = snap.embeddings(model)
    v = np.zeros((len(ids), emb.shape[1]), dtype=np.float32)
    for a in range(0, len(rows), 8192):  # the memory-mapped vectors, a block at a time
        np.add.at(v, chunk_doc[a:a + 8192], np.asarray(emb[rows[a:a + 8192]], dtype=np.float32))
    return ids, v / np.linalg.norm(v, axis=1, keepdims=True)


def project(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """PCA and t-SNE coordinates, and the share of the variance the two principal components keep."""
    from sklearn.manifold import TSNE
    centred = v - v.mean(axis=0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    reduced = centred @ vt[:PCA_DIMS_FOR_TSNE].T
    tsne = TSNE(2, perplexity=30, init="pca", learning_rate="auto", random_state=0, n_jobs=-1).fit_transform(reduced)
    return reduced[:, :2].copy(), tsne.astype(np.float32), float(var[:2].sum())


def normalise(xy: np.ndarray) -> np.ndarray:
    xy = xy - xy.mean(axis=0)
    return xy / np.sqrt((xy ** 2).sum(axis=1).mean())


def align(xy: np.ndarray, ids: list[int], ref_xy: np.ndarray, ref_ids: list[int]) -> tuple[np.ndarray, int]:
    """xy turned to match ref_xy on the shared decisions (orthogonal Procrustes with scale); also how many are shared."""
    ref_pos = {d: i for i, d in enumerate(ref_ids)}
    shared = [(i, ref_pos[d]) for i, d in enumerate(ids) if d in ref_pos]
    xy = normalise(xy)
    if len(shared) < 10:
        return xy, len(shared)
    a, r = xy[[i for i, _ in shared]], ref_xy[[j for _, j in shared]]
    mu_a, mu_r = a.mean(axis=0), r.mean(axis=0)
    u, s, vt = np.linalg.svd((a - mu_a).T @ (r - mu_r))
    rot = u @ vt
    scale = s.sum() / ((a - mu_a) ** 2).sum()
    return ((xy - mu_a) @ rot * scale + mu_r).astype(np.float32), len(shared)


def nearest(v: np.ndarray, k: int = NEIGHBOURS) -> tuple[np.ndarray, np.ndarray]:
    """The k most similar decisions of each (cosine, full space), most similar first."""
    idx = np.empty((len(v), k), dtype=np.int32)
    cos = np.empty((len(v), k), dtype=np.float32)
    for a in range(0, len(v), 2048):
        sims = v[a:a + 2048] @ v.T
        sims[np.arange(len(sims)), np.arange(a, a + len(sims))] = -np.inf
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_sims, axis=1)
        idx[a:a + 2048] = np.take_along_axis(top, order, axis=1)
        cos[a:a + 2048] = np.take_along_axis(top_sims, order, axis=1)
    return idx, cos


def citation_stats(snap: e.Snapshot, ids: list[int], v: np.ndarray, nn: np.ndarray, rng) -> dict:
    """Are decisions linked by a citation closer than random pairs? Links to the Garante's own procedural rules are
    left out, as in viz_embeddings.py."""
    pos = {d: i for i, d in enumerate(ids)}
    procedural = {pos[d] for d in ids if viz.PROCEDURAL_TYPE.lower() in ", ".join(snap.docs[d]["tipologia"]).lower()}
    edges = np.array([(i, j) for i, j in viz.citation_edges(ids, pos, snap.path)
                      if i not in procedural and j not in procedural]).reshape(-1, 2)
    if not len(edges):
        return {"links": 0}
    linked = (v[edges[:, 0]] * v[edges[:, 1]]).sum(axis=1)
    pairs = rng.integers(0, len(ids), (viz.N_RANDOM_PAIRS, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    random_ = (v[pairs[:, 0]] * v[pairs[:, 1]]).sum(axis=1)
    both = np.concatenate([edges, edges[:, ::-1]])  # each link seen from both ends
    hits = (nn[both[:, 0]] == both[:, 1, None]).any(axis=1)
    return {"links": int(len(edges)), "auc": viz.auc(linked, random_), "recall10": float(hits.mean()),
            "median_linked": float(np.median(linked)), "median_random": float(np.median(random_))}


def write_docs(snap: e.Snapshot) -> None:
    docs = [{"id": d["docweb_id"], "t": d["title"], "d": d["date"], "u": d["url"], "ty": d["tipologia"],
             "a": d["argomenti"], "n": len(snap.similar.get(d["docweb_id"], []))} for d in snap.docs.values()]
    docs_path(snap.name).write_text(json.dumps(docs, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def build() -> None:
    MAPS.mkdir(exist_ok=True)
    index, refs = {}, {}
    rng = np.random.default_rng(0)
    order = [e.NOW] + [s for s in e.SNAPSHOT_LABELS if s != e.NOW]  # the reference first
    for name in order:
        snap = e.open_snapshot(name)
        write_docs(snap)
        models = sorted(snap.models(), key=lambda m: m != REFERENCE[1])
        for model in models:
            for scope in SCOPES:
                t = time.time()
                ids, v = doc_vectors(snap, model, scope)
                if len(ids) < 50:  # v0 has no sections, so no violation passages
                    continue
                pca, tsne, var2d = project(v)
                shared = {}
                coords = {}
                for proj, xy in (("pca", pca), ("tsne", tsne)):
                    if (name, model) == REFERENCE:
                        coords[proj] = refs[(scope, proj)] = normalise(xy).astype(np.float32)
                        refs[(scope, "ids")] = ids
                        shared[proj] = len(ids)
                    else:
                        coords[proj], shared[proj] = align(xy, ids, refs[(scope, proj)], refs[(scope, "ids")])
                nn_idx, nn_cos = nearest(v)
                key = map_key(name, model, scope)
                np.savez_compressed(MAPS / f"{key}.npz", ids=np.array(ids, dtype=np.int64), pca=coords["pca"],
                                    tsne=coords["tsne"], nn_idx=nn_idx, nn_cos=nn_cos.astype(np.float16))
                index[key] = {"snapshot": name, "model": model, "scope": scope, "documents": len(ids),
                              "passages": int(b.scope_mask(snap.chunks, scope).sum()), "var2d": var2d,
                              "shared_with_reference": shared["pca"],
                              **citation_stats(snap, ids, v, nn_idx, rng)}
                s = index[key]
                print(f"{key:48s} {len(ids):6d} decisions  PCA var(2D) {var2d:.1%}  citation AUC "
                      f"{s.get('auc', float('nan')):.3f}  recall@10 {s.get('recall10', float('nan')):.1%}  "
                      f"({time.time() - t:.0f} s)", flush=True)
        INDEX.write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"{len(index)} maps -> {MAPS}")


if __name__ == "__main__":
    build()
