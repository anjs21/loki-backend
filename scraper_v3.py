"""
Scraper con filtro per data per recuperare documenti mancanti 2018-2019.
Usa l'URL Liferay con lifecycle=0 e parametri dataInizio/dataFine.

Utilizzo:
  python scraper_v3.py
  python scraper_v3.py --data-inizio 2018-01-01 --data-fine 2019-12-31
"""

import argparse
import csv
import re
import time
import urllib.parse
from datetime import timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL    = "https://www.garanteprivacy.it"
PORTLET_ID  = "g_gpdp5_search_GGpdp5SearchPortlet"
NS          = f"_{PORTLET_ID}_"
EXISTING_CSVS = ["provvedimenti.csv", "scraper_v2.csv"]
OUTPUT_FILE  = "scraper_v3.csv"
DELAY_SECONDS = 2

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "it-IT,it;q=0.9,en;q=0.8",
}

FIELDNAMES = ["titolo", "data", "tipologia", "argomenti", "estratto", "url_documento", "docweb_id"]


def load_existing_ids() -> set:
    ids = set()
    for fp in EXISTING_CSVS:
        if not Path(fp).exists():
            continue
        with open(fp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("docweb_id"):
                    ids.add(row["docweb_id"])
    return ids


def build_url(data_inizio: str, data_fine: str, page: int = 1) -> str:
    params = {
        "p_p_id": PORTLET_ID,
        "p_p_lifecycle": "0",
        "p_p_state": "normal",
        "p_p_mode": "view",
        f"{NS}mvcRenderCommandName": "/renderSearch",
        f"{NS}dataInizio": data_inizio,
        f"{NS}dataFine": data_fine,
        f"{NS}cur": page,
    }
    return f"{BASE_URL}/home/ricerca?" + urllib.parse.urlencode(params)


def get_page(url: str, session: requests.Session) -> BeautifulSoup | None:
    try:
        r = session.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        return BeautifulSoup(r.text, "html.parser")
    except requests.RequestException as e:
        print(f"    [ERRORE] {url}: {e}")
        return None


def get_total_pages(soup: BeautifulSoup) -> int:
    pagination = soup.find("ul", class_="pagination")
    if not pagination:
        return 1
    nums = [
        int(a.get_text(strip=True))
        for a in pagination.find_all("a", class_="page-link")
        if a.get_text(strip=True).isdigit()
    ]
    return max(nums) if nums else 1


def extract_docweb_id(href: str) -> str:
    m = re.search(r"/docweb/(\d+)", href)
    return m.group(1) if m else ""


def parse_card(card) -> dict:
    result = {k: "" for k in FIELDNAMES}

    title_link = card.find("a", class_="titolo-risultato")
    if title_link:
        result["titolo"] = title_link.get_text(strip=True)
        href = title_link.get("href", "")
        if href:
            result["url_documento"] = href if href.startswith("http") else BASE_URL + href
            result["docweb_id"] = extract_docweb_id(href)

    date_div = card.find("div", class_="data-risultato")
    if date_div:
        p = date_div.find("p")
        if p:
            result["data"] = p.get_text(strip=True)

    estratto_p = card.find("p", class_="estratto-risultato")
    if estratto_p:
        result["estratto"] = estratto_p.get_text(strip=True)

    result["tipologia"] = "; ".join(
        t for link in card.find_all("a", href=re.compile(r"/search/tipologia/"))
        if (t := link.get_text(strip=True))
    )
    result["argomenti"] = "; ".join(
        t for link in card.find_all("a", href=re.compile(r"/search/argomento/"))
        if (t := link.get_text(strip=True))
    )
    return result


def fmt(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def scrape_range(data_inizio: str, data_fine: str, existing_ids: set, session: requests.Session) -> list[dict]:
    print(f"\nRange: {data_inizio} → {data_fine}")

    soup = get_page(build_url(data_inizio, data_fine, 1), session)
    if not soup:
        print("  [SKIP] impossibile caricare la prima pagina")
        return []

    cards_first = soup.find_all("div", class_="card-risultato")
    if not cards_first:
        print("  [SKIP] nessuna card in p.1")
        return []

    total_pages = get_total_pages(soup)
    print(f"  Pagine: {total_pages}  (~{total_pages*10:,} documenti totali)")

    results = []
    nuovi = 0
    skip = 0
    start_t = time.time()

    for page_num in range(1, total_pages + 1):
        if page_num == 1:
            s = soup
        else:
            url = build_url(data_inizio, data_fine, page_num)
            s = None
            for attempt in range(1, 4):
                s = get_page(url, session)
                if s:
                    cards = s.find_all("div", class_="card-risultato")
                    if cards:
                        break
                    print(f"    [warn] p.{page_num} vuota, retry {attempt}/3...")
                    time.sleep(10)
            if not s:
                print(f"    [SKIP] p.{page_num} non scaricata")
                time.sleep(DELAY_SECONDS)
                continue

        cards = s.find_all("div", class_="card-risultato")
        if not cards:
            print(f"    [warn] p.{page_num} senza card, salto")
            time.sleep(DELAY_SECONDS)
            continue

        for card in cards:
            r = parse_card(card)
            if not r["docweb_id"]:
                continue
            if r["docweb_id"] in existing_ids:
                skip += 1
            else:
                results.append(r)
                existing_ids.add(r["docweb_id"])
                nuovi += 1

        if page_num % 20 == 0:
            elapsed = time.time() - start_t
            rate = page_num / elapsed
            remaining = (total_pages - page_num) / rate if rate > 0 else 0
            print(f"    p.{page_num}/{total_pages} | nuovi: {nuovi} | skip: {skip} | ETA: {fmt(remaining)}")

        if page_num < total_pages:
            time.sleep(DELAY_SECONDS)

    elapsed = time.time() - start_t
    print(f"  Completato in {fmt(elapsed)} — nuovi: {nuovi}, già presenti: {skip}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-inizio", default="2018-01-01")
    parser.add_argument("--data-fine", default="2019-12-31")
    args = parser.parse_args()

    print(f"Carico ID esistenti da {EXISTING_CSVS}...")
    existing_ids = load_existing_ids()
    print(f"  {len(existing_ids):,} docweb_id già presenti\n")

    session = requests.Session()
    results = scrape_range(args.data_inizio, args.data_fine, existing_ids, session)

    print(f"\n{'='*60}")
    print(f"TOTALE NUOVI RECORD: {len(results):,}")
    print("=" * 60)

    if not results:
        print("Nessun nuovo record trovato.")
        return

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    print(f"Salvati in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
