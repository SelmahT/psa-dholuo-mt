"""
PSA Source Collector — raw download only, no cleaning/EDA.

What this does:
  - Visits each source's start URL
  - Also follows a handful of internal links that look like news/press/
    campaign pages (homepages alone are mostly navigation, not content)
  - Pulls out ALL visible body text per page (not just headings/paragraphs)
  - Saves ONE ROW PER PAGE into data/raw/<source_name>.csv (raw, as-is)
  - Saves everything combined into data/raw/all_sources_raw.csv (raw, as-is)
  - ALSO saves a sentence-organized version into
    data/raw/all_sources_sentences.csv — raw text split one sentence per
    row, with common site boilerplate (cookie notices, "skip to content",
    copyright footers, nav labels, bare dates) filtered out and exact
    repeats removed. This is still NOT translation cleaning — no rewording,
    no topic filtering, just removing obvious non-content junk so the file
    is usable. The raw per-source CSVs and all_sources_raw.csv are
    untouched and keep 100% of the original text, boilerplate included.

What this does NOT do:
  - No deduplication
  - No cleaning/normalizing text
  - No translation alignment (English/Kiswahili/Dholuo are NOT split out —
    that's manual/Trizzah's job, this just grabs raw page text as-is)
  - No topic classification beyond the tag you assign in SOURCES below

Install requirements first:
    pip install requests beautifulsoup4 lxml pypdf --break-system-packages

Run:
    python collect_psa_sources.py

Add more URLs any time by editing the SOURCES list below.
"""

import csv
import os
import re
import time
import datetime
import requests
import urllib3
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

# We deliberately retry with verify=False for a couple of known-flaky .go.ke
# sites (see fetch_page_text). Silence the resulting warning so it doesn't
# spam the console — the fallback is intentional, not accidental.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

OUTPUT_DIR = "data/raw"
REQUEST_DELAY_SECONDS = 2  # be polite to servers
TIMEOUT = 20
MAX_LINKS_PER_SOURCE = 6  # how many extra internal pages to follow per source

# Link text/URL keywords that suggest an actual content page (as opposed to
# "About Us", "Contact", "Careers", etc.)
CONTENT_KEYWORDS = [
    "news", "press", "statement", "campaign", "alert", "notice",
    "media", "article", "blog", "story", "update", "bulletin",
    "publication", "announcement", "health", "advisory", "release",
]

MAX_RETRIES = 2  # extra attempts for connection-level failures (not HTTP errors)
RETRY_BACKOFF_SECONDS = 3

# Boilerplate that shows up on nearly every page and isn't PSA content —
# used ONLY to filter the sentence-organized CSV. The raw per-source CSVs
# are untouched and keep everything, including this text.
NOISE_SUBSTRINGS = [
    "skip to content", "all rights reserved", "cookie", "privacy policy",
    "terms of service", "terms and conditions", "subscribe to our newsletter",
    "follow us on", "read more", "load more", "have an existing account",
    "sign in", "log in", "sign up", "tweets by", "open twitter",
    "connect with us", "share this", "related posts", "leave a comment",
    "powered by", "designed by", "ruby design", "foxiz news network",
    "back to top", "page not found", "404",
]
# Lines that are ONLY a date (e.g. "Friday, 10 Jul 2026", "July 10, 2026")
DATE_ONLY_RE = re.compile(
    r"^(?:\w+,\s*)?\d{1,2}\s+\w+\s+\d{4}$|^\w+\s+\d{1,2},\s*\d{4}$",
    re.IGNORECASE,
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36"
}

# ---------------------------------------------------------------------------
# SOURCE LIST — add/remove rows here. "domain_tag" = your project's topic
# (Agriculture / Governance / Health / Security / Education). "source_name"
# controls which CSV file the row lands in.
# ---------------------------------------------------------------------------
SOURCES = [
    # ==== PATRICIA'S 11 CONFIRMED SOURCES (from PSA_Content_Sources_Log.xlsx) ====

    # 1. Ramogi FM — Dholuo, streaming only, no archive
    {"source_name": "ramogi_fm", "domain_tag": "Mixed",
     "url": "https://royalmedia.co.ke/brands/ramogi-fm/"},

    # 2. KBC — state broadcaster, vernacular/Dholuo service
    # NOTE: Patricia's sheet gives no URL for this one — kbc.co.ke is my
    # best-guess homepage, NOT confirmed by her. Verify before relying on it.
    {"source_name": "kbc", "domain_tag": "Mixed",
     "url": "https://www.kbc.co.ke/"},

    # 3. Ministry of Health
    {"source_name": "moh", "domain_tag": "Health",
     "url": "https://www.health.go.ke/"},

    # 4. IEBC — Voter Education Curriculum
    {"source_name": "iebc", "domain_tag": "Governance",
     "url": "https://www.iebc.or.ke/"},

    # 5. NSDCC — HIV/TB/malaria awareness, also active on X
    {"source_name": "nsdcc", "domain_tag": "Health",
     "url": "https://nsdcc.go.ke/"},

    # 6. NACADA — drug/alcohol abuse prevention
    {"source_name": "nacada", "domain_tag": "Security",
     "url": "https://nacada.go.ke/"},

    # 7. Kenya Red Cross Society (KRCS) — disaster/health alerts
    {"source_name": "krcs", "domain_tag": "Mixed",
     "url": "https://redcross.or.ke/"},

    # 8. UNICEF Kenya — child health/nutrition/education
    {"source_name": "unicef_kenya", "domain_tag": "Health",
     "url": "https://www.unicef.org/kenya/"},

    # 9. WHO Kenya / WHO AFRO
    {"source_name": "who_kenya", "domain_tag": "Health",
     "url": "https://www.afro.who.int/countries/kenya"},

    # 10. NTSA — road safety ("Usalama Barabarani")
    {"source_name": "ntsa", "domain_tag": "Security",
     "url": "https://www.ntsa.go.ke/"},

    # 11. Radio Nam Lolwe FM — Dholuo, Kisumu, streams online
    {"source_name": "nam_lolwe_fm", "domain_tag": "Mixed",
     "url": "https://radionamlolwefm.com/"},
]


def fetch_url(url: str) -> tuple[requests.Response, str]:
    """GET a URL, with SSL fallback for known-flaky gov sites, and a couple
    of retries for transient connection failures (timeouts, DNS hiccups,
    connection reset) — some sites just need a second try."""
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp, ""
        except requests.exceptions.SSLError:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False)
            resp.raise_for_status()
            return resp, "SSL verification skipped (site has cert issues)"
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = e
            if attempt < MAX_RETRIES:
                print(f"    retrying ({attempt + 1}/{MAX_RETRIES}) after: {e}")
                time.sleep(RETRY_BACKOFF_SECONDS)
    raise last_error


def extract_body_text(soup: BeautifulSoup) -> str:
    """Grab ALL visible text on the page, not just headings/paragraphs/list
    items — modern gov/news sites put real content inside plain <div>/<span>
    wrappers that a tag-restricted extractor misses entirely."""
    # Strip elements that are never real content
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer",
                      "svg", "form", "iframe"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse repeated blank lines / whitespace without rewording anything
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def find_content_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    """Find internal links whose URL or link text suggests an actual content
    page (news/press/campaign/etc), so we don't just sit on the homepage."""
    base_domain = urlparse(base_url).netloc
    found = []
    seen = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
            continue
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.netloc != base_domain:
            continue  # stay on the same site
        link_text = a.get_text(strip=True).lower()
        haystack = (full_url + " " + link_text).lower()
        if any(kw in haystack for kw in CONTENT_KEYWORDS):
            if full_url not in seen:
                seen.add(full_url)
                found.append(full_url)
        if len(found) >= MAX_LINKS_PER_SOURCE:
            break
    return found


def fetch_page(url: str) -> tuple[str, str, list[str]]:
    """Download a URL, return (raw_text, note, discovered_links). No text
    cleaning beyond whitespace collapse."""
    resp, note = fetch_url(url)

    content_type = resp.headers.get("Content-Type", "")
    if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
        return extract_pdf_text(resp.content), note, []

    soup = parse_html(resp.text)
    links = find_content_links(soup, url)
    return extract_body_text(soup), note, links


def parse_html(html: str) -> BeautifulSoup:
    """Use lxml if available, otherwise fall back to Python's built-in
    parser so a missing/broken lxml install never blocks collection."""
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def is_noise(sentence: str) -> bool:
    """True if this looks like site chrome rather than real content.
    Used ONLY when building the sentence-organized CSV — never touches the
    raw per-source CSVs."""
    s = sentence.strip()
    low = re.sub(r"\s+", " ", s.lower())
    if len(s.split()) < 3:
        return True
    if any(phrase in low for phrase in NOISE_SUBSTRINGS):
        return True
    if DATE_ONLY_RE.match(s):
        return True
    # Mostly non-alphabetic (icons/symbols/decorative unicode headers)
    letters = sum(ch.isalpha() for ch in s)
    if letters < len(s) * 0.5:
        return True
    return False


def dedupe_preserve_order(sentences: list[str]) -> list[str]:
    """Drop exact repeats within the same page (e.g. a headline that appears
    both in a teaser block and again in a "related articles" list). Only
    used for the sentence CSV."""
    seen = set()
    out = []
    for s in sentences:
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def split_into_sentences(text: str) -> list[str]:
    """Mechanical sentence split on ./!/? — no NLP, no rewording, just
    breaking raw text into rows. Filters out fragments under 3 words
    (menu items, single labels) so the sentence CSV isn't mostly noise."""
    # Treat each line as its own boundary too (site text is often already
    # broken into short blocks that don't end in punctuation)
    sentences = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        for piece in SENTENCE_SPLIT_RE.split(block):
            piece = piece.strip()
            if len(piece.split()) >= 3:
                sentences.append(piece)
    return sentences


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract raw text from a PDF's bytes. Requires pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[PDF TEXT EXTRACTION SKIPPED — run: pip install pypdf --break-system-packages]"

    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_rows = []
    by_source = {}
    sentence_rows = []

    for entry in SOURCES:
        source_name = entry["source_name"]
        domain_tag = entry["domain_tag"]
        start_url = entry["url"]

        urls_to_fetch = [start_url]
        visited = set()

        while urls_to_fetch:
            url = urls_to_fetch.pop(0)
            if url in visited:
                continue
            visited.add(url)

            print(f"Fetching [{source_name}] {url} ...")
            try:
                raw_text, note, links = fetch_page(url)
                status = "OK" if not note else f"OK ({note})"
                # Only queue newly discovered links when we're still on the
                # start page, so we don't cascade into a deep crawl
                if url == start_url:
                    for link in links:
                        if link not in visited:
                            urls_to_fetch.append(link)
            except Exception as e:
                raw_text = ""
                status = f"ERROR: {e}"
                print(f"  -> failed: {e}")

            row = {
                "source_name": source_name,
                "domain_tag": domain_tag,
                "url": url,
                "date_collected": datetime.date.today().isoformat(),
                "status": status,
                "raw_text": raw_text,
            }
            all_rows.append(row)
            by_source.setdefault(source_name, []).append(row)

            page_sentences = split_into_sentences(raw_text)
            page_sentences = [s for s in page_sentences if not is_noise(s)]
            page_sentences = dedupe_preserve_order(page_sentences)
            for i, sentence in enumerate(page_sentences):
                sentence_rows.append({
                    "source_name": source_name,
                    "domain_tag": domain_tag,
                    "url": url,
                    "date_collected": row["date_collected"],
                    "sentence_index": i,
                    "sentence": sentence,
                })

            time.sleep(REQUEST_DELAY_SECONDS)

    # One CSV per source (raw, as-is)
    for source_name, rows in by_source.items():
        path = os.path.join(OUTPUT_DIR, f"{source_name}.csv")
        write_csv(path, rows, ["source_name", "domain_tag", "url", "date_collected", "status", "raw_text"])
        print(f"Saved {len(rows)} row(s) -> {path}")

    # Combined CSV, still raw/page-level
    combined_path = os.path.join(OUTPUT_DIR, "all_sources_raw.csv")
    write_csv(combined_path, all_rows, ["source_name", "domain_tag", "url", "date_collected", "status", "raw_text"])
    print(f"Saved {len(all_rows)} row(s) -> {combined_path}")

    # Combined CSV, sentence-organized (one sentence per row)
    sentences_path = os.path.join(OUTPUT_DIR, "all_sources_sentences.csv")
    write_csv(sentences_path, sentence_rows,
              ["source_name", "domain_tag", "url", "date_collected", "sentence_index", "sentence"])
    print(f"Saved {len(sentence_rows)} row(s) -> {sentences_path}")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
