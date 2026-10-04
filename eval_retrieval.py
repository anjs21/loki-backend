#!/usr/bin/env python3
"""
eval_retrieval.py - compare embedding models on the POC index using the Garante's topic tags as weak labels.

A retrieved decision counts as relevant when its `argomenti` include one of the query's tags. Tags are assigned
by the Garante's editors, are not exhaustive (about 4% of decisions have none) and are topical rather than about
the specific violation, so treat the scores as a relative comparison between models, not absolute quality.

  python eval_retrieval.py                          # all models with embeddings, default hybrid weight
  python eval_retrieval.py --alpha 0 0.3 0.5 1      # compare embeddings only / hybrid / BM25 only
  python eval_retrieval.py --models jasper -v       # one model, print each query's top results

Near-duplicate template families are indexed once, so a query whose relevant decisions are one family (e.g.
"revenge porn") can reach at most P@10 = 0.1; read MRR for those.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import build_index as b

# (query, relevant tags). Queries avoid repeating the tag wording where possible, to test meaning over keywords.
QUERIES = [
    ("telecamere che riprendono i lavoratori sul posto di lavoro", {"Videosorveglianza"}),
    ("violazione dei dati personali a seguito di attacco informatico", {"Data breach", "Cybersecurity"}),
    ("chiamate promozionali verso numeri iscritti al registro delle opposizioni",
     {"Registro delle opposizioni", "Telemarketing", "Telefonate promozionali"}),
    ("rimozione dai risultati di ricerca di articoli su vicende giudiziarie passate",
     {"Diritto all'oblio", "Motori di ricerca"}),
    ("pubblicazione di immagini intime senza il consenso della persona ritratta", {"revenge porn"}),
    ("richiesta di accesso civico a documenti che contengono dati personali di terzi", {"Accesso civico"}),
    ("dossier sanitario consultato da personale non autorizzato",
     {"Dossier sanitario", "Fascicolo sanitario elettronico", "Aziende sanitarie", "Dati sanitari"}),
    ("cookie e tracciamento degli utenti sui siti web", {"Cookies", "Profilazione"}),
    ("email pubblicitarie inviate senza consenso",
     {"E-mail promozionali", "Marketing", "Comunicazioni indesiderate", "Consenso"}),
    ("designazione del responsabile della protezione dei dati negli enti pubblici", {"Responsabile protezione dati"}),
    ("valutazione d'impatto prima di avviare un nuovo trattamento ad alto rischio", {"DPIA"}),
    ("sistemi di intelligenza artificiale e chatbot che trattano dati personali", {"Intelligenza artificiale"}),
    ("dati sulle vaccinazioni e green pass durante la pandemia", {"Coronavirus"}),
    ("pubblicazione online delle graduatorie dei concorsi pubblici",
     {"Graduatorie di concorsi", "Pubblicazioni online", "Trasparenza amministrativa"}),
    ("dati reddituali dei contribuenti nell'anagrafe tributaria",
     {"Agenzia delle Entrate", "Anagrafe tributaria", "Fisco", "Dichiarazioni dei redditi"}),
    # English queries over Italian documents (cross-lingual)
    ("CCTV cameras monitoring employees", {"Videosorveglianza"}),
    ("ransomware attack exposing patient data", {"Data breach", "Cybersecurity"}),
    ("right to be forgotten and delisting from search engines", {"Diritto all'oblio", "Motori di ricerca", "Google"}),
    ("unsolicited telemarketing calls",
     {"Telemarketing", "Telefonate promozionali", "Comunicazioni indesiderate", "Registro delle opposizioni"}),
    ("AI chatbot processing personal data", {"Intelligenza artificiale"}),
    ("anti-money laundering checks by banks", {"Antiriciclaggio", "Banche credito e finanza"}),
    ("non-consensual sharing of intimate images", {"revenge porn"}),
]
K = 10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", choices=b.MODELS,
                    default=[m for m in b.MODELS if b.embeddings_path(m).exists()])
    ap.add_argument("--alpha", type=float, nargs="+", default=[b.HYBRID_ALPHA],
                    help="BM25 share(s) in the fused ranking; 0 = embeddings only, 1 = BM25 only")
    ap.add_argument("--no-weights", action="store_true", help="plain cosine, without section weights")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    chunks = [json.loads(line) for line in (b.OUT / "chunks.jsonl").open(encoding="utf-8")]
    for c in chunks:
        c.pop("embed_text", None)
    weights = None if args.no_weights else np.array([c.get("weight", 1.0) for c in chunks], dtype=np.float32)
    bm25 = b.BM25() if any(a > 0 for a in args.alpha) else None
    results = {}
    for key in args.models:
        emb = np.load(b.embeddings_path(key)).astype(np.float32)
        assert len(emb) == len(chunks), f"{key}: {len(emb)} embeddings for {len(chunks)} chunks"
        model = b.load_model(key)
        q_emb = b.encode_queries(model, key, [q for q, _ in QUERIES])
        for alpha in args.alpha:
            if alpha >= 1 and any(n.startswith("bm25") for n in results):
                continue  # BM25 alone does not depend on the embedding model
            name = "bm25" if alpha >= 1 else f"{key}" if alpha <= 0 else f"{key}+bm25 {alpha:g}"
            rows = []
            for (query, tags), q in zip(QUERIES, q_emb):
                scores = b.score_chunks(q, query, emb, bm25, weights, alpha)
                ranked = b.rank_documents(scores, chunks, K)
                rel = [bool(tags & set(chunks[h["top"][1]]["argomenti"])) for h in ranked.values()]
                p_at_k = sum(rel) / K
                rr = next((1 / (i + 1) for i, r in enumerate(rel) if r), 0.0)
                rows.append((query, p_at_k, rr))
                if args.verbose:
                    print(f"\n[{name}] {query}  P@{K}={p_at_k:.1f}")
                    for r, h in zip(rel, ranked.values()):
                        c = chunks[h["top"][1]]
                        print(f"   {'✓' if r else '·'} {h['top'][0]:.4g} {c['title'][:70]} | "
                              f"{', '.join(c['argomenti'][:3])}")
            results[name] = rows
        del model

    width = max(16, *(len(k) + 2 for k in results))
    print(f"\n{'query':<72}" + "".join(f"{k:>{width}}" for k in results))
    for i, (query, _) in enumerate(QUERIES):
        print(f"{query[:70]:<72}" + "".join(f"{f'P {results[k][i][1]:.1f}  RR {results[k][i][2]:.2f}':>{width}}"
                                             for k in results))
    print("-" * (72 + width * len(results)))
    for name, sl in (("Italian queries", slice(0, 15)), ("English queries", slice(15, None)), ("All", slice(None))):
        line = f"{name + f' — mean P@{K} / MRR':<72}"
        for k, rows in results.items():
            sub = rows[sl]
            line += f"{f'{np.mean([r[1] for r in sub]):.3f} / {np.mean([r[2] for r in sub]):.3f}':>{width}}"
        print(line)


if __name__ == "__main__":
    main()
