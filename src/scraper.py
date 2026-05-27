"""
scraper.py  (Curated URL version)
----------------------------------
Scrapes a fixed list of high-quality pages from japan.travel.
No crawling — we target exactly the pages we want.
"""

import os
import json
import time
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DELAY      = float(os.getenv("REQUEST_DELAY", 1.5))
OUTPUT_DIR = "data/raw"

# ── Load URLs from config file ────────────────────────────────────────────────
def load_urls(filepath: str = "data/urls.txt") -> list:
    """
    Load URLs from a text file.
    Ignores empty lines and lines starting with #.
    Deduplicates automatically.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"URL list not found: {filepath}")
 
    urls = []
    seen = set()
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if line not in seen:
                    urls.append(line)
                    seen.add(line)
 
    print(f"📋 Loaded {len(urls)} URLs from {filepath}")
    return urls


# ── Content extraction ────────────────────────────────────────────────────────

def extract_text(html: str) -> str:
    """
    Extract clean article content using targeted CSS selectors.
    Targets japan.travel's content container specifically.
    """
    soup = BeautifulSoup(html, "lxml")
 
    # Remove scripts and noise tags first
    for tag in soup(["script", "style", "noscript", "iframe"]):
        tag.decompose()
 
    # Target the main content container using CSS selectors
    content_area = (
        soup.select_one("div.content-main-wrapper")
        or soup.select_one("div.mod-wysiwyg__body")
        or soup.select_one("div.split__left")
        or soup.body
    )
 
    if not content_area:
        return ""
 
    # Remove remaining noise sections inside the content area
    for noise in content_area.select(
        "div.mod-header, div.mod-breadcrumb, "
        "div.otChoicesBnr, div.mod-footer-container, "
        "div.mod-web-survey-footer__container, "
        "div.mod-loc-container, div._l-box-content_1bi77_161"
    ):
        noise.decompose()
 
    # Extract headings, paragraphs and list items
    tags = content_area.find_all(["h1", "h2", "h3", "h4", "p", "li"])
 
    lines = []
    seen = set()
 
    for tag in tags:
        text = tag.get_text(separator=" ", strip=True)
 
        if len(text) < 40:
            continue
        if text in seen:
            continue
        if "${" in text:                    # unrendered JS template
            continue
        words = text.split()
        if len(words) > 0 and text.count(",") / len(words) > 0.3:
            continue                        # language/country list
 
        seen.add(text)
        lines.append(text)
 
    return "\n\n".join(lines)


# ── Main scraper ──────────────────────────────────────────────────────────────

def scrape():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    urls = load_urls("data/urls.txt")  

    saved  = 0
    failed = 0
    index  = []

    print(f"🚀 Japan Travel — Curated scraper")
    print(f"   Pages to scrape : {len(urls)}")
    print(f"   Delay           : {DELAY}s\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()

        # Accept cookies once — browser context remembers it for all pages
        cookies_accepted = False

        for url in tqdm(urls, desc="Scraping"):
            try:
                page.goto(url, timeout=20000)
                page.wait_for_load_state("networkidle", timeout=15000)

                # Dismiss cookie popup on first page only
                if not cookies_accepted:
                    try:
                        page.click("button:has-text('Accept All Cookies')", timeout=4000)
                        page.wait_for_timeout(800)
                        cookies_accepted = True
                    except:
                        pass

                # Dismiss chatbot
                try:
                    page.keyboard.press("Escape")
                    page.wait_for_timeout(300)
                except:
                    pass

            except PlaywrightTimeout:
                tqdm.write(f"  ⏱  Timeout: {url}")
                failed += 1
                continue
            except Exception as e:
                tqdm.write(f"  ❌ Error: {url} — {e}")
                failed += 1
                continue

            title = page.title() or "No title"

            # Skip 404 pages
            if "not found" in title.lower():
                tqdm.write(f"  🚫 404: {url}")
                failed += 1
                continue

            html  = page.content()
            text  = extract_text(html)

            if len(text) < 200:
                tqdm.write(f"  ⚠️  Too short ({len(text)} chars): {title}")
                failed += 1
                continue

            data = {
                "url":        url,
                "title":      title,
                "content":    text,
                "word_count": len(text.split()),
            }

            filename = f"page_{saved:03d}.json"
            with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            index.append({
                "file":       filename,
                "url":        url,
                "title":      title,
                "word_count": data["word_count"],
            })

            tqdm.write(f"  ✅ [{data['word_count']:>4}w] {title[:60]}")
            saved += 1
            time.sleep(DELAY)

        browser.close()

    # Save index
    with open("data/scrape_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    total_words = sum(p["word_count"] for p in index)
    print(f"\n{'─'*50}")
    print(f"✅  Saved   : {saved}/{len(urls)} pages")
    print(f"🚫  Skipped : {failed} pages")
    print(f"📝  Words   : {total_words:,} total")
    print(f"{'─'*50}")


if __name__ == "__main__":
    scrape()