#!/usr/bin/env python3
"""
garante_scraper.py - polite, resumable scraper for Garante Privacy "Provvedimenti" (docweb).

Stages (run in order; each is resumable and safe to re-run):
  listing  crawl search result pages  -> doc IDs + listing metadata (SQLite)
  docs     fetch every docweb page     -> raw HTML, gzipped, in data/raw/docs/
  parse    parse raw HTML offline      -> data/docs.jsonl
  stats    corpus statistics           -> stdout + data/stats.json

Raw HTML is always kept, so a parser fix never requires re-scraping.

Setup:
  pip install httpx beautifulsoup4 lxml
  export SCRAPER_CONTACT="you@your-university.eu"   # identifies you to the site operator

Smoke test first (a few minutes), then inspect data/stats.json and a couple of raw files:
  python garante_scraper.py listing --end 2
  python garante_scraper.py docs --limit 20
  python garante_scraper.py parse
  python garante_scraper.py stats

Full backfill (~15k requests; ~10 h at the default 2 s delay):
  python garante_scraper.py listing
  python garante_scraper.py docs
Daily sync afterwards:
  python garante_scraper.py listing --order DESC --stop-when-known && python garante_scraper.py docs
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

BASE = "https://www.garanteprivacy.it"
SEARCH_URL = f"{BASE}/home/ricerca"
DOC_URL = BASE + "/home/docweb/-/docweb-display/docweb/{id}"
P = "_g_gpdp5_search_GGpdp5SearchPortlet_"
# The 31 "tipologia" IDs behind the site's own "Provvedimenti" filter (copied from its pagination links).
TIPOLOGIA_IDS = ("9445099,9567234,10484,10485,2034210,2024563,2034211,9660377,10488,9150852,"
                 "10492,10498,10499,10500,10503,9161733,10516,10526,10527,10528,10529,10530,"
                 "10532,9150851,2010735,9271403,9625871,10535,9126615,10546,10533")

DATA = Path(os.environ.get("GARANTE_DATA", "data"))
CONTACT = os.environ.get("SCRAPER_CONTACT", "").strip()
DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
UA = os.environ.get("SCRAPER_USER_AGENT", DEFAULT_UA)
MIN_DELAY = float(os.environ.get("SCRAPER_DELAY", "2.0"))  # seconds between requests

DOCWEB_HREF = re.compile(r"/docweb/-/docweb-display/docweb/(\d+)")
DOCJSP_HREF = re.compile(r"doc\.jsp\?ID=(\d+)")
DOCWEB_TEXT = re.compile(r"doc\.?\s*web\s*n\.?\s*(\d{5,9})", re.I)
# "euro 24.000 (ventiquattromila)", "euro 20.000,00 (ventimila)"; the words double-check the digits.
FINE = re.compile(r"euro\s+(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{2}))?\s*\(\s*([a-zàèéìòù ]+?)\s*\)", re.I)
REGISTRO = re.compile(r"Registro\s+dei\s+provvedimenti\s+n\.\s*(\d+)\s+del\s+([^\n]{6,30}?\d{4})", re.I)
SECTION_STOP = re.compile(r"^(Vedi anche|Documenti citati|Footer)", re.I)
NOISE_LINES = {"Ascolta", "Stampa", "Stampa Stampa", "Condividi", "Menù azioni",
               "e-mail", "facebook", "linkedin", "twitter"}
GDPR_START = date(2018, 5, 25)


# ----------------------------------------------------------------------------- storage
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DATA / "state.sqlite")
    con.executescript("""
        CREATE TABLE IF NOT EXISTS listing_pages(
            order_dir TEXT, page INTEGER, n_items INTEGER, fetched_at TEXT,
            PRIMARY KEY (order_dir, page));
        CREATE TABLE IF NOT EXISTS docs(
            docweb_id INTEGER PRIMARY KEY, title TEXT, list_date TEXT, tipologia TEXT,
            argomenti TEXT, listing_text TEXT, first_seen TEXT,
            fetch_status INTEGER, fetched_at TEXT, raw_sha256 TEXT);
    """)
    return con


def save_raw(kind: str, name: str, content: bytes) -> None:
    folder = DATA / "raw" / kind
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(gzip.compress(content))


def load_raw(kind: str, name: str) -> str:
    return gzip.decompress((DATA / "raw" / kind / name).read_bytes()).decode("utf-8", errors="replace")


# ----------------------------------------------------------------------------- polite HTTP
class Fetcher:
    def __init__(self) -> None:
        headers = {
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
        }
        if CONTACT:
            headers["X-Scraper-Contact"] = CONTACT
        self.client = httpx.Client(
            http1=True,
            http2=False,
            headers=headers,
            timeout=60,
            follow_redirects=True
        )
        self.robots = RobotFileParser()
        try:
            r = self.client.get(BASE + "/robots.txt")
            if r.status_code == 200:
                self.robots.parse(r.text.splitlines())
        except Exception as e:
            print(f"[info] Could not parse robots.txt directly ({e}); proceeding with defaults.")
        self.delay = max(MIN_DELAY, float(self.robots.crawl_delay(UA) or 0))
        self._last = 0.0
        print(f"[info] {self.delay:.1f}s between requests, UA: {UA}")

    def get(self, url: str, params: dict | None = None) -> httpx.Response:
        full = str(httpx.URL(url, params=params))
        if self.robots.entries and not self.robots.can_fetch(UA, full):
            raise PermissionError(f"robots.txt disallows {full}")
        for attempt in range(6):
            wait = self.delay * random.uniform(1.0, 1.5) - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.client.get(url, params=params)
                if r.status_code in (200, 404):
                    return r
                problem = f"HTTP {r.status_code}"
                if r.headers.get("Retry-After", "").isdigit():
                    time.sleep(int(r.headers["Retry-After"]))
            except httpx.TransportError as e:
                problem = repr(e)
            backoff = min(600, 10 * 2 ** attempt)
            print(f"[retry {attempt + 1}/6] {problem} - sleeping {backoff}s")
            time.sleep(backoff)
        raise RuntimeError(f"Giving up on {full}")


# ----------------------------------------------------------------------------- helpers
def _text(el) -> str:
    return el.get_text(" ", strip=True) if el is not None else ""


def _labels(links: list[Tag], kind: str) -> list[str]:
    """Facet labels from links like /search/argomento/Videosorveglianza."""
    return list(dict.fromkeys(_text(a) for a in links if f"/search/{kind}/" in a.get("href", "") and _text(a)))


def _ids(links: list[Tag]) -> list[int]:
    out = []
    for a in links:
        m = DOCWEB_HREF.search(a.get("href", "")) or DOCJSP_HREF.search(a.get("href", ""))
        if m:
            out.append(int(m.group(1)))
    return list(dict.fromkeys(out))


def _iso_date(s: str | None) -> str | None:
    m = re.fullmatch(r"(\d{2})/(\d{2})/(\d{2}|\d{4})", (s or "").strip())
    if not m:
        return None
    d, mth, y = map(int, m.groups())
    if y < 100:  # the "Scheda" box uses 2-digit years; the Garante started in 1997
        y += 1900 if y >= 90 else 2000
    try:
        return date(y, mth, d).isoformat()
    except ValueError:
        return None


def _euro(m: re.Match) -> float:
    return float(m.group(1).replace(".", "") + "." + (m.group(2) or "00"))


# ----------------------------------------------------------------------------- listing
def listing_params(page: int, order: str) -> dict:
    params = {"p_p_id": "g_gpdp5_search_GGpdp5SearchPortlet", "p_p_lifecycle": "0",
              "p_p_state": "normal", "p_p_mode": "view"}
    fields = {"mvcRenderCommandName": "/renderSearch", "text": "", "dataInizio": "", "dataFine": "",
              "idsTipologia": TIPOLOGIA_IDS, "idsArgomenti": "", "quanteParole": "",
              "quanteParoleStr": "", "nonParoleStr": "", "paginaWeb": "false", "allegato": "false",
              "ordinamentoPer": order, "ordinamentoTipo": "data", "cur": str(page)}
    params.update({P + k: v for k, v in fields.items()})
    return params


def _result_box(a: Tag, title_ids: set[int]) -> Tag | None:
    """Largest ancestor of a result title that contains no other result title."""
    best = None
    for parent in a.parents:
        if parent.name in ("body", "html", "[document]"):
            break
        if sum(1 for x in parent.find_all("a", href=DOCWEB_HREF) if id(x) in title_ids) > 1:
            break
        best = parent
    return best


def parse_listing(html: str) -> tuple[int | None, list[dict]]:
    soup = BeautifulSoup(html, "lxml")
    m = re.search(r"(\d[\d.]*)\s*risultati di ricerca", soup.get_text(" "))
    total = int(m.group(1).replace(".", "")) if m else None
    anchors = soup.find_all("a", href=DOCWEB_HREF)
    # Result titles are wrapped in <strong>; this excludes navigation links such as "L'agenda".
    # (Not all titles end with "[<docweb id>]": long ones are truncated with "...", and some have no id.)
    titles = [a for a in anchors if a.parent is not None and a.parent.name == "strong"]
    if not titles:
        titles = [a for a in anchors if not a.find_parent(["nav", "header", "footer"])]
    title_ids = {id(a) for a in titles}
    items = []
    for a in titles:
        box = _result_box(a, title_ids)
        links = box.find_all("a", href=True) if box is not None else []
        box_text = _text(box)
        d = re.search(r"\b(\d{2}/\d{2}/\d{4})\b", box_text)
        items.append({"docweb_id": int(DOCWEB_HREF.search(a["href"]).group(1)),
                      "title": _text(a),
                      "list_date": d.group(1) if d else None,
                      "tipologia": _labels(links, "tipologia"),
                      "argomenti": _labels(links, "argomento"),
                      "listing_text": box_text[:600]})
    return total, items


def cmd_listing(args) -> None:
    con, f = db(), Fetcher()
    order = args.order
    if args.start:
        page = args.start
    elif args.stop_when_known:
        page = 1
    else:  # resume a backfill where it stopped
        page = con.execute("SELECT COALESCE(MAX(page), 0) FROM listing_pages WHERE order_dir=?",
                           (order,)).fetchone()[0] + 1
    last_page = args.end
    while last_page is None or page <= last_page:
        r = f.get(SEARCH_URL, params=listing_params(page, order))
        if r.status_code != 200:
            print(f"[stop] HTTP {r.status_code} on page {page}")
            break
        save_raw("listing", f"{order}_{page:05d}.html.gz", r.content)
        total, items = parse_listing(r.text)
        if not items:
            print(f"[stop] no results parsed on page {page} (end of results, or the page layout changed)")
            break
        if last_page is None and total:
            last_page = math.ceil(total / len(items))
            print(f"[info] {total} results -> {last_page} pages")
        new = 0
        for it in items:
            known = con.execute("SELECT 1 FROM docs WHERE docweb_id=?", (it["docweb_id"],)).fetchone()
            new += known is None
            con.execute("""
                INSERT INTO docs(docweb_id, title, list_date, tipologia, argomenti, listing_text, first_seen)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(docweb_id) DO UPDATE SET title=excluded.title, list_date=excluded.list_date,
                    tipologia=excluded.tipologia, argomenti=excluded.argomenti,
                    listing_text=excluded.listing_text""",
                (it["docweb_id"], it["title"], it["list_date"], json.dumps(it["tipologia"], ensure_ascii=False),
                 json.dumps(it["argomenti"], ensure_ascii=False), it["listing_text"], now()))
        con.execute("INSERT OR REPLACE INTO listing_pages VALUES (?,?,?,?)", (order, page, len(items), now()))
        con.commit()
        dates = [it["list_date"] for it in items if it["list_date"]]
        print(f"[listing {order}] page {page}/{last_page}: {len(items)} items, {new} new, "
              f"dates {dates[0] if dates else '?'} .. {dates[-1] if dates else '?'}")
        if args.stop_when_known and new == 0:
            print("[stop] no new documents on this page")
            break
        page += 1


# ----------------------------------------------------------------------------- documents
def cmd_docs(args) -> None:
    con, f = db(), Fetcher()
    order = "RANDOM()" if args.random else "docweb_id"
    todo = [row[0] for row in con.execute(
        "SELECT docweb_id FROM docs WHERE fetch_status IS NULL OR fetch_status NOT IN (200, 404, -1) "
        f"ORDER BY {order}")]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[info] {len(todo)} documents to fetch (~{len(todo) * f.delay * 1.25 / 3600:.1f} h)")
    for i, doc_id in enumerate(todo, 1):
        status, sha = None, None
        try:
            r = f.get(DOC_URL.format(id=doc_id))
            status = r.status_code
            if status == 200:
                save_raw("docs", f"{doc_id}.html.gz", r.content)
                sha = hashlib.sha256(r.content).hexdigest()
        except PermissionError as e:
            print(f"[skip] {e}")
            status = -1
        con.execute("UPDATE docs SET fetch_status=?, fetched_at=?, raw_sha256=? WHERE docweb_id=?",
                    (status, now(), sha, doc_id))
        con.commit()
        if i % 50 == 0 or i == len(todo):
            print(f"[docs] {i}/{len(todo)}")


# ----------------------------------------------------------------------------- parsing
def _heading(soup: BeautifulSoup, pattern: str) -> Tag | None:
    rx = re.compile(pattern, re.I)
    return next((h for h in soup.find_all(["h2", "h3", "h4"]) if rx.match(_text(h))), None)


def _section(heading: Tag | None) -> tuple[list[Tag], str]:
    """Links and text from a heading up to the next 'Vedi anche' / 'Documenti citati' / 'Footer' heading."""
    if heading is None:
        return [], ""
    links, parts = [], []
    for el in heading.next_elements:
        if isinstance(el, Tag):
            if el is not heading and el.name in ("h1", "h2", "h3") and SECTION_STOP.match(_text(el)):
                break
            if el.name == "a" and el.get("href"):
                links.append(el)
        elif type(el) is NavigableString:
            parts.append(str(el))
    return links, " ".join(" ".join(parts).split())


def _body(soup: BeautifulSoup, doc_id: int, title: str) -> tuple[str, str]:
    """Decision text: from '[doc. web n. <id>]' to the 'Scheda' metadata box."""
    lines = [" ".join(ln.split()) for ln in soup.get_text("\n").splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    end_m = re.search(r"^Scheda$", text, re.M) or re.search(r"^Footer$", text, re.M)
    end = end_m.start() if end_m else len(text)
    m = re.search(rf"\[\s*doc\.?\s*web\s*n\.?\s*{doc_id}\s*\]", text, re.I)
    if m and m.start() < end:
        return text[m.start():end].strip(), "docweb_marker"
    i = text.rfind(title, 0, end) if title else -1  # fallback: after the in-page title
    if i >= 0:
        chunk = "\n".join(ln for ln in text[i + len(title):end].splitlines() if ln.strip() not in NOISE_LINES)
        return chunk.strip(), "title_fallback"
    return "", "failed"


def parse_doc(doc_id: int, html: str, listing: dict) -> dict:
    soup = BeautifulSoup(html, "lxml")
    og = soup.find("meta", attrs={"property": "og:title"})
    title = (og.get("content") or "").strip() if og else ""
    if not title and soup.title:
        title = _text(soup.title).removesuffix(" - Garante Privacy")
    robots = soup.find("meta", attrs={"name": "robots"})
    scheda_links, scheda_text = _section(_heading(soup, r"^Scheda$"))
    see_also_links, _ = _section(_heading(soup, r"^Vedi anche"))
    cited_links, _ = _section(_heading(soup, r"^Documenti citati"))
    body, method = _body(soup, doc_id, title)
    d = re.search(r"Data\s+(\d{2}/\d{2}/\d{2,4})", scheda_text)
    iso = _iso_date(listing.get("list_date")) or _iso_date(d.group(1) if d else None)
    reg = REGISTRO.search(body)
    return {
        "docweb_id": doc_id,
        "url": DOC_URL.format(id=doc_id),
        "title": title,
        "date": iso,
        # Decision date only. Which law applies (old Codice vs GDPR) is a later extraction step.
        "era": None if not iso else ("pre_gdpr" if date.fromisoformat(iso) < GDPR_START else "gdpr"),
        "tipologia": _labels(scheda_links, "tipologia") or json.loads(listing.get("tipologia") or "[]"),
        "argomenti": _labels(scheda_links, "argomento") or json.loads(listing.get("argomenti") or "[]"),
        "registro_n": int(reg.group(1)) if reg else None,
        "fines_eur": sorted({_euro(m) for m in FINE.finditer(body)}),  # heuristic: verify before display
        "cites": _ids(cited_links),
        "see_also": _ids(see_also_links),
        "refs_in_text": sorted({int(x) for x in DOCWEB_TEXT.findall(body)} - {doc_id}),
        "attachments": sorted({a["href"] for a in soup.find_all("a", href=True)
                               if "/documents/" in a["href"]
                               and re.search(r"\.(pdf|docx?|rtf|odt|zip)(\?|$)", a["href"], re.I)}),
        "noarchive": bool(robots and "noarchive" in (robots.get("content") or "").lower()),
        "xx_placeholders": len(re.findall(r"\bXX\b", body)),
        "body_method": method,
        "body_chars": len(body),
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),  # use this, not raw_sha256, to detect edits
        "body": body,
    }


def cmd_parse(args) -> None:
    con = db()
    rows = con.execute("SELECT docweb_id, list_date, tipologia, argomenti FROM docs WHERE fetch_status=200").fetchall()
    out, errors = DATA / "docs.jsonl", []
    with out.open("w", encoding="utf-8") as fh:
        for doc_id, list_date, tip, arg in rows:
            try:
                rec = parse_doc(doc_id, load_raw("docs", f"{doc_id}.html.gz"),
                                {"list_date": list_date, "tipologia": tip, "argomenti": arg})
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception as e:  # keep going; report at the end
                errors.append((doc_id, repr(e)))
    print(f"[parse] {len(rows) - len(errors)} parsed -> {out}; {len(errors)} errors")
    for e in errors[:10]:
        print("   ", e)


# ----------------------------------------------------------------------------- stats
def cmd_stats(args) -> None:
    path = DATA / "docs.jsonl"
    recs = [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []
    if not recs:
        sys.exit("No parsed documents yet; run `parse` first.")
    lengths = sorted(r["body_chars"] for r in recs)
    pct = lambda p: lengths[min(len(lengths) - 1, int(p / 100 * len(lengths)))]
    fined = [max(r["fines_eur"]) for r in recs if r["fines_eur"]]
    chunks = sum(math.ceil(n / 1600) for n in lengths)  # ~400-token chunks at ~4 chars/token
    stats = {
        "documents": len(recs),
        "body_method": Counter(r["body_method"] for r in recs),
        "suspect_short_bodies_lt_300_chars": sum(1 for r in recs if r["body_chars"] < 300),
        "body_chars_p10_p50_p90_max": [pct(10), pct(50), pct(90), lengths[-1]],
        "approx_tokens_total": sum(lengths) // 4,
        "est_chunks_400tok": chunks,
        "est_vector_mb_1024d_halfvec": round(chunks * 1024 * 2 / 1e6, 1),
        "missing_date": sum(1 for r in recs if not r["date"]),
        "by_year": dict(sorted(Counter((r["date"] or "????")[:4] for r in recs).items())),
        "era": Counter(r["era"] for r in recs),
        "tipologia": Counter(t for r in recs for t in r["tipologia"]).most_common(),
        "argomenti_top40": Counter(t for r in recs for t in r["argomenti"]).most_common(40),
        "docs_with_fine": len(fined),
        "fine_eur_median_max": [statistics.median(fined), max(fined)] if fined else None,
        "docs_with_attachments": sum(1 for r in recs if r["attachments"]),
        "noarchive_pages": sum(1 for r in recs if r["noarchive"]),
        "docs_with_XX_pseudonyms": sum(1 for r in recs if r["xx_placeholders"]),
        "citation_edges": sum(len(set(r["cites"]) | set(r["refs_in_text"])) for r in recs),
    }
    (DATA / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in stats.items():
        print(f"{k}: {v}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("listing", help="crawl search result pages")
    p.add_argument("--order", choices=["ASC", "DESC"], default="ASC",
                   help="ASC (oldest first) keeps pages stable while new decisions are published")
    p.add_argument("--start", type=int)
    p.add_argument("--end", type=int)
    p.add_argument("--stop-when-known", action="store_true",
                   help="incremental sync (use with --order DESC): stop at the first page with no new IDs")
    p = sub.add_parser("docs", help="fetch docweb pages")
    p.add_argument("--limit", type=int)
    p.add_argument("--random", action="store_true", help="random order (useful for a varied smoke test)")
    sub.add_parser("parse", help="parse raw HTML -> data/docs.jsonl")
    sub.add_parser("stats", help="corpus statistics -> data/stats.json")
    args = ap.parse_args()
    {"listing": cmd_listing, "docs": cmd_docs, "parse": cmd_parse, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main()
