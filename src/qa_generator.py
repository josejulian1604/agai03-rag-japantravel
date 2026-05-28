"""
qa_generator.py
---------------
Reads scraped pages from data/raw/ and uses Claude API to generate
synthetic Q/A pairs for each page. Saves results to data/qa_dataset.csv.

Output columns: question, answer, source_url, source_title
"""

import os
import json
import time
import csv
import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR      = "data/raw"
OUTPUT_CSV   = "data/qa_dataset.csv"
TARGET_PAIRS = 150          # minimum Q/A pairs we want in total
DELAY        = 1.0          # seconds between API calls (rate limit safety)
MODEL        = "claude-sonnet-4-6"

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(title: str, content: str, num_pairs: int) -> str:
    """
    Build the prompt that asks Claude to generate Q/A pairs.
    We ask for JSON output so it's easy to parse.
    """
    return f"""You are building a Q/A dataset for a Japan travel chatbot.

Given the following travel content, generate exactly {num_pairs} question-answer pairs.

Rules:
- Questions must be natural, specific, and directly answerable from the text
- Answers must be factual, concise (1-3 sentences), and based ONLY on the provided content
- Cover different aspects of the content (don't repeat similar questions)
- Questions should sound like real traveler queries
- Do NOT include questions about links, images, or website navigation

Content title: {title}
Content:
{content[:3000]}

Respond ONLY with a valid JSON array, no explanation, no markdown:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""


# ── API call ──────────────────────────────────────────────────────────────────
def generate_qa_pairs(title: str, content: str, num_pairs: int) -> list:
    """
    Call Claude API and parse the JSON response.
    Returns a list of dicts: [{question, answer}, ...]
    """
    prompt = build_prompt(title, content, num_pairs)

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()

        # Clean up common JSON issues
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        pairs = json.loads(raw)

        # Validate structure
        valid = []
        for pair in pairs:
            if isinstance(pair, dict) and "question" in pair and "answer" in pair:
                q = pair["question"].strip()
                a = pair["answer"].strip()
                if len(q) > 10 and len(a) > 10:
                    valid.append({"question": q, "answer": a})

        return valid

    except json.JSONDecodeError as e:
        print(f"\n  ⚠️  JSON parse error: {e}")
        return []
    except Exception as e:
        print(f"\n  ❌ API error: {e}")
        return []
    
def pairs_for_page(word_count: int) -> int:
    """
    Adaptive Q/A pairs based on content length.
    Short pages get fewer pairs to avoid hallucination and redundancy.
    """
    if word_count < 200:
        return 2
    elif word_count < 400:
        return 3
    elif word_count < 700:
        return 5
    elif word_count < 1200:
        return 7
    else:
        return 9


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load all scraped pages
    pages = []
    for fname in sorted(os.listdir(RAW_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(RAW_DIR, fname), encoding="utf-8") as f:
                pages.append(json.load(f))

    if not pages:
        print("❌ No pages found in data/raw/. Run scraper.py first.")
        return

    print(f"📄 Found {len(pages)} scraped pages")

    # 3. Generate Q/A pairs for each page
    all_pairs = []

    for page in tqdm(pages, desc="Generating Q/A"):
        title       = page.get("title", "")
        content     = page.get("content", "")
        url         = page.get("url", "")
        word_count  = page.get("word_count", 0)

        if len(content) < 100:
            tqdm.write(f"  ⚠️  Skipping (too short): {title}")
            continue
        
        num_pairs = pairs_for_page(word_count)
        pairs = generate_qa_pairs(title, content, num_pairs)

        for pair in pairs:
            all_pairs.append({
                "question":     pair["question"],
                "answer":       pair["answer"],
                "source_url":   url,
                "source_title": title,
            })

        tqdm.write(f"  ✅ [{len(pairs):>2} pairs] {title[:55]}")
        time.sleep(DELAY)

    # 4. Save to CSV
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "answer", "source_url", "source_title"]
        )
        writer.writeheader()
        writer.writerows(all_pairs)

    print(f"\n{'─'*55}")
    print(f"✅  Total Q/A pairs generated : {len(all_pairs)}")
    print(f"📁  Saved to                  : {OUTPUT_CSV}")
    print(f"{'─'*55}")

    # 5. Preview first 5 pairs
    print("\n📋 Sample Q/A pairs:")
    for i, pair in enumerate(all_pairs[:5], 1):
        print(f"\n  [{i}] Q: {pair['question']}")
        print(f"      A: {pair['answer'][:100]}...")


if __name__ == "__main__":
    main()