#!/usr/bin/env python3
"""
upload_supabase.py - load the search index (data/poc, built by build_index.py) into Supabase:

* the database (Postgres): the decisions with their full text, the chunks with their vectors, the word statistics
  and which model built the vectors. The tables come from loki-backend/supabase/migrations (Supabase CLI).
* Storage: the files the search API ranks with, in memory (loki-backend/app/search.py): every chunk's vector, its
  BM25 weights, section and decision. They go in the public bucket `search-index`, in a folder named after the
  index id, which the database records in index_meta; the API's build downloads that folder
  (loki-backend/scripts/fetch_index.py). After an upload, redeploy the API to give it the new index.

  # three settings: the database connection, the project URL and its secret key (dashboard -> Project Settings ->
  # API Keys -> Secret keys). Both secrets give full access: type them in the shell, never save them in the repo.
  read -rs DATABASE_URL && export DATABASE_URL           # hosted: the "Session pooler" connection string
  export SUPABASE_URL=https://<project ref>.supabase.co
  read -rs SUPABASE_SECRET_KEY && export SUPABASE_SECRET_KEY
  python upload_supabase.py
  python upload_supabase.py --dry-run          # counts and sizes only, uploads nothing

  # local Supabase (loki-backend: npx supabase start; the secret key is in npx supabase status)
  DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres SUPABASE_URL=http://127.0.0.1:54321 \\
    SUPABASE_SECRET_KEY=sb_secret_... python upload_supabase.py

The Storage files go first, in a new folder; then the tables are emptied, and reloaded in a second transaction that
also records the new index id in index_meta; then older index folders are deleted. The old rows are deleted first
because the database can't hold two copies: emptying and reloading in one transaction keeps the old copy on disk
until the end, and with the write-ahead log that filled the Free plan's 2 GB disk (Postgres then shuts down). So while
it runs the search API has no data, and a failed run leaves the tables empty, or half loaded with no index_id (the
API refuses to start): run it again.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np

import build_index as b

HEADER_LABELS = re.compile(r"\b(Data|Tipologia|Argomenti|Sezione|Paragrafo): ")
DECISION_COLUMNS = ["id", "title", "date", "url", "types", "topics", "similar_decisions", "body"]
CHUNK_COLUMNS = ["id", "decision_id", "idx", "section", "heading", "weight", "year", "types", "is_case", "header",
                 "text", "embedding"]
BUCKET = "search-index"
INDEX_FORMAT = 1                  # loki-backend/app/search.py reads this format
MAX_FILE_BYTES = 45_000_000       # Storage takes at most 50 MB a file on the Free plan


def load(model: str):
    chunks = [json.loads(line) for line in (b.OUT / "chunks.jsonl").open(encoding="utf-8")]
    emb = np.load(b.embeddings_path(model))
    if len(emb) != len(chunks):
        sys.exit(f"{b.embeddings_path(model)} has {len(emb)} vectors for {len(chunks)} chunks: re-run the embed stage")
    ids = {c["docweb_id"] for c in chunks}
    similar = json.loads((b.OUT / "duplicates.json").read_text(encoding="utf-8"))
    decisions = []
    for line in (b.DATA / "docs.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["docweb_id"] in ids:
            decisions.append({
                "id": r["docweb_id"], "title": r["title"], "url": r["url"],
                "date": date.fromisoformat(r["date"]) if r["date"] else None,
                "types": r["tipologia"], "topics": r["argomenti"],
                "similar_decisions": [{"id": t["docweb_id"], "date": t["date"], "title": t["title"], "url": t["url"]}
                            for t in similar.get(str(r["docweb_id"]), [])],
                "body": "\n".join(b.paragraphs(r["body"])),  # Italian text only, broken lines rejoined
            })
    if len(decisions) != len(ids):
        sys.exit(f"{len(ids) - len(decisions)} decisions of the index are missing from {b.DATA / 'docs.jsonl'}")
    return decisions, chunks, emb


def chunk_row(c: dict, vec) -> list:
    header = HEADER_LABELS.sub("", c["embed_text"].split("\n\n", 1)[0])
    return [c["chunk_id"], c["docweb_id"], c["chunk_index"], c["section"], c.get("heading", ""), c["weight"],
            int(c["date"][:4]) if c["date"] else None, c["tipologia"], not b.CASE_TYPES.isdisjoint(c["tipologia"]),
            header, c["text"], vec]


def index_id(model: str) -> str:
    """Identifies this build of the index: a hash of the chunks, their vectors and the BM25 index."""
    h = hashlib.sha256()
    for path in (b.OUT / "chunks.jsonl", b.embeddings_path(model), b.OUT / "bm25.npz", b.OUT / "bm25_vocab.json"):
        with path.open("rb") as f:
            while block := f.read(1 << 20):
                h.update(block)
    return h.hexdigest()[:16]


def write_search_index(out: Path, ident: str, model: str, decisions: list[dict], chunks: list[dict], emb) -> list[Path]:
    """The files of loki-backend/app/search.py (see its docstring) for this index, in `out`; manifest.json last."""
    bm25 = b.BM25()  # checks that the BM25 index matches chunks.jsonl
    decisions = sorted(decisions, key=lambda d: d["id"])
    row = {d["id"]: i for i, d in enumerate(decisions)}
    types = sorted({t for d in decisions for t in d["types"]})
    type_code = {t: i for i, t in enumerate(types)}
    sections = sorted({c["section"] for c in chunks})
    section_code = {s: i for i, s in enumerate(sections)}
    meta = {
        "sections": sections, "types": types,
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "chunk_decision": [row[c["docweb_id"]] for c in chunks],
        "chunk_section": [section_code[c["section"]] for c in chunks],
        "chunk_weight": [c.get("weight", 1.0) for c in chunks],
        "decision_ids": [d["id"] for d in decisions],
        "decision_year": [d["date"].year if d["date"] else None for d in decisions],
        "decision_types": [[type_code[t] for t in d["types"]] for d in decisions],
        "decision_is_case": [not b.CASE_TYPES.isdisjoint(d["types"]) for d in decisions],
        "decision_similar": [len(d["similar_decisions"]) for d in decisions],
        "vocab": bm25.vocab,
    }
    files = [out / "meta.json.gz"]
    files[0].write_bytes(gzip.compress(json.dumps(meta, ensure_ascii=False).encode("utf-8")))
    (out / "bm25.npz").write_bytes((b.OUT / "bm25.npz").read_bytes())
    files.append(out / "bm25.npz")
    rows = MAX_FILE_BYTES // (emb.shape[1] * emb.itemsize)
    vectors = []
    for part, start in enumerate(range(0, len(emb), rows)):
        path = out / f"vectors-{part}.npy"
        np.save(path, np.ascontiguousarray(emb[start:start + rows], dtype=np.float16))
        vectors.append(path.name)
        files.append(path)
    for f in files:
        if f.stat().st_size > MAX_FILE_BYTES:
            sys.exit(f"{f.name} is {f.stat().st_size / 1e6:.0f} MB, over Storage's 50 MB a file: split it")
    spec = b.MODELS[model]
    manifest = {
        "format": INDEX_FORMAT, "index_id": ident, "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": {"key": model, "name": spec["name"], "revision": spec.get("revision", "")},
        "dimensions": int(emb.shape[1]), "chunks": len(chunks), "decisions": len(decisions),
        "analyzer": {"stemmer": "italian", "word_pattern": b._WORD.pattern, "stopwords": sorted(b.STOPWORDS_IT)},
        "vectors": vectors,
        "files": {f.name: {"bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
                  for f in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return files + [out / "manifest.json"]


class Storage:
    """Supabase Storage, with the project's secret key: the public bucket of the search index files."""

    def __init__(self, url: str, key: str) -> None:
        import httpx
        # A secret key (sb_secret_...) goes in apikey only; a legacy service_role key is a JWT and also in Authorization.
        headers = {"apikey": key, **({"Authorization": f"Bearer {key}"} if key.startswith("eyJ") else {})}
        self.http = httpx.Client(base_url=f"{url.rstrip('/')}/storage/v1", headers=headers, timeout=600)

    def _check(self, r, what: str):
        if r.status_code >= 400:
            sys.exit(f"Storage: {what} failed: {r.status_code} {r.text[:300]}")
        return r

    def ensure_bucket(self) -> None:
        r = self.http.get(f"/bucket/{BUCKET}")
        if r.status_code == 200:
            if not r.json().get("public"):
                self._check(self.http.put(f"/bucket/{BUCKET}", json={"public": True}), "making the bucket public")
            return
        self._check(self.http.post("/bucket", json={"id": BUCKET, "name": BUCKET, "public": True}), "creating the bucket")

    def upload(self, folder: str, files: list[Path]) -> None:
        for f in files:
            with f.open("rb") as body:
                # A folder per index id is never overwritten with other content, so caches can keep the files.
                self._check(self.http.post(f"/object/{BUCKET}/{folder}/{f.name}", content=body, headers={
                    "content-type": "application/octet-stream", "cache-control": "max-age=31536000",
                    "x-upsert": "true"}), f"uploading {f.name}")
            print(f"  {folder}/{f.name} ({f.stat().st_size / 1e6:.1f} MB)", flush=True)

    def remove_folders_except(self, keep: str) -> list[str]:
        def listing(prefix: str) -> list[dict]:
            return self._check(self.http.post(f"/object/list/{BUCKET}", json={"prefix": prefix, "limit": 1000}),
                               "listing the bucket").json()
        removed = []
        for entry in listing(""):
            if entry.get("id") is None and entry["name"] != keep:  # a folder: an older index
                paths = [f"{entry['name']}/{f['name']}" for f in listing(entry["name"])]
                if paths:
                    self._check(self.http.request("DELETE", f"/object/{BUCKET}", json={"prefixes": paths}),
                                f"deleting {entry['name']}")
                removed.append(entry["name"])
        return removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=b.MODELS, default=b.DEFAULT_MODEL)
    ap.add_argument("--dry-run", action="store_true", help="read and check the index, print sizes, write nothing")
    args = ap.parse_args()
    decisions, chunks, emb = load(args.model)
    text_mb = (sum(len(d["body"].encode()) for d in decisions) + sum(len(c["text"].encode()) for c in chunks)) / 1e6
    print(f"{len(decisions)} decisions, {len(chunks)} chunks, vectors {emb.shape[1]}-d; "
          f"text {text_mb:.0f} MB, vectors {len(chunks) * emb.shape[1] * 2 / 1e6:.0f} MB as halfvec")
    settings = {k: os.environ.get(k) for k in ("DATABASE_URL", "SUPABASE_URL", "SUPABASE_SECRET_KEY")}
    missing = [k for k, v in settings.items() if not v]
    if missing and not args.dry_run:
        sys.exit(f"Set {', '.join(missing)} (see the top of this file).")
    ident = index_id(args.model)
    with tempfile.TemporaryDirectory() as tmp:
        files = write_search_index(Path(tmp), ident, args.model, decisions, chunks, emb)
        print(f"search index {ident}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.0f} MB")
        if args.dry_run:
            return
        storage = Storage(settings["SUPABASE_URL"], settings["SUPABASE_SECRET_KEY"])
        storage.ensure_bucket()
        storage.upload(ident, files)
    load_database(settings["DATABASE_URL"], args.model, ident, decisions, chunks, emb)
    removed = storage.remove_folders_except(ident)
    if removed:
        print(f"older search indexes deleted from Storage: {', '.join(removed)}")
    print("Now redeploy the search API, so it downloads this index.")


def load_database(url: str, model: str, ident: str, decisions: list[dict], chunks: list[dict], emb) -> None:
    import psycopg
    from psycopg.types.json import Jsonb
    from pgvector import HalfVector
    from pgvector.psycopg import register_vector
    t0 = time.time()
    with psycopg.connect(url) as conn:
        register_vector(conn)
        conn.execute("set statement_timeout = 0")
        # Committed on its own, so the old rows' disk space is free before the new ones are written.
        conn.execute("truncate public.chunks, public.decisions, public.lexeme_stats, public.index_meta")
        conn.commit()
        print("old index deleted", flush=True)
        with conn.cursor() as cur:
            with cur.copy(f"copy public.decisions ({', '.join(DECISION_COLUMNS)}) from stdin") as copy:
                for d in decisions:
                    copy.write_row([Jsonb(d[k]) if k == "similar_decisions" else d[k] for k in DECISION_COLUMNS])
            print(f"decisions loaded ({time.time() - t0:.0f}s)", flush=True)
            with cur.copy(f"copy public.chunks ({', '.join(CHUNK_COLUMNS)}) from stdin with (format binary)") as copy:
                copy.set_types(["text", "bigint", "int4", "text", "text", "float4", "int4", "text[]", "bool", "text",
                                "text", "halfvec"])
                for i, (c, vec) in enumerate(zip(chunks, emb)):
                    copy.write_row(chunk_row(c, HalfVector(vec)))
                    if (i + 1) % 10000 == 0:
                        print(f"  chunks {i + 1}/{len(chunks)} ({time.time() - t0:.0f}s)", flush=True)
            print(f"chunks loaded ({time.time() - t0:.0f}s)", flush=True)
            cur.execute("insert into public.lexeme_stats select word, ndoc from ts_stat("
                        "'select to_tsvector(''italian'', header || '' '' || text) from public.chunks')")
            cur.execute("select count(*) from public.lexeme_stats")
            print(f"{cur.fetchone()[0]} word stems counted ({time.time() - t0:.0f}s)")
            spec = b.MODELS[model]
            meta = json.loads((b.OUT / "chunks_meta.json").read_text(encoding="utf-8"))
            cur.executemany("insert into public.index_meta (key, value) values (%s, %s)", [
                ("model", model), ("model_name", spec["name"]), ("model_revision", spec.get("revision", "")),
                ("dimensions", str(emb.shape[1])), ("decisions", str(len(decisions))), ("chunks", str(len(chunks))),
                ("source_sha256", meta["source_sha256"]), ("uploaded_at", time.strftime("%Y-%m-%d %H:%M:%S")),
                ("index_id", ident),  # the search API's files in Storage (write_search_index)
            ])
        conn.commit()
        conn.autocommit = True
        conn.execute("analyze public.decisions, public.chunks, public.lexeme_stats")
        size = conn.execute("select pg_size_pretty(pg_total_relation_size('public.decisions')), "
                            "pg_size_pretty(pg_total_relation_size('public.chunks')), "
                            "pg_size_pretty(pg_database_size(current_database()))").fetchone()
    print(f"done in {time.time() - t0:.0f}s: decisions {size[0]}, chunks {size[1]}, whole database {size[2]}")


if __name__ == "__main__":
    main()
