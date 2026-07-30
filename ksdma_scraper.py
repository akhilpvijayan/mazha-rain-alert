"""
KSDMA Rainfall Alert Scraper
============================
Fetches https://sdma.kerala.gov.in/rainfall-2/, extracts Red/Orange/Yellow
rainfall alerts grouped by district, and reports any NEW alerts since the
last run (so you can wire this into a push-notification pipeline).

Pipeline:
  1. Fetch page HTML
  2. Try Gemini LLM extraction first (handles messy/inconsistent formatting)
  3. Fall back to Groq if Gemini fails or isn't configured
  4. Fall back to regex extraction if both LLMs fail
  5. Validate districts against known Kerala district list
  6. Diff against last saved state -> report new alerts
  7. Save current state for next run

Usage:
  export GEMINI_API_KEY="your_key_here"   # primary  - aistudio.google.com
  export GROQ_API_KEY="your_key_here"     # fallback - console.groq.com/keys
  python ksdma_scraper.py

Run this on a schedule (cron / GitHub Actions / cloud function) every 1-2 hours.

---
Bugs found and fixed against REAL captured data from the live page (not
guessed - each one below broke actual production runs before being fixed):

1. Districts sometimes wrap to the line AFTER "date :" instead of being on
   the same line -> handled by Case 3 in extract_with_regex.
2. An alert-level header (e.g. "ഓറഞ്ച് അലർട്ട്") can appear on the SAME line
   as the first date entry -> a level-keyword match no longer skips the
   rest of that line's processing (Case 2 / embedded-date handling).
3. A date/number can get split mid-character across two lines by the page's
   HTML (e.g. "30/07/2026" becomes a line "3" then "0/07/2026: ...") ->
   fixed by rejoin_broken_number_lines(), called from extract_main_text().
4. Colors must be filtered to a specific date (defaults to today) - a
   5-day forecast blended together would show today as an alert color
   that's actually only valid for a future day.
"""

import os
import re
import json
import sys
from datetime import datetime, timezone, timedelta

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

KSDMA_URL = "https://sdma.kerala.gov.in/rainfall-2/"
STATE_FILE = os.path.join(os.path.dirname(__file__), "last_state.json")
COLORS_FILE = os.path.join(os.path.dirname(__file__), "district_colors.json")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

IST = timezone(timedelta(hours=5, minutes=30))

# Malayalam -> English district lookup (all 14 Kerala districts)
DISTRICT_MAP = {
    "തിരുവനന്തപുരം": "Thiruvananthapuram",
    "കൊല്ലം": "Kollam",
    "പത്തനംതിട്ട": "Pathanamthitta",
    "ആലപ്പുഴ": "Alappuzha",
    "കോട്ടയം": "Kottayam",
    "ഇടുക്കി": "Idukki",
    "എറണാകുളം": "Ernakulam",
    "തൃശൂർ": "Thrissur",
    "പാലക്കാട്": "Palakkad",
    "മലപ്പുറം": "Malappuram",
    "കോഴിക്കോട്": "Kozhikode",
    "വയനാട്": "Wayanad",
    "കണ്ണൂർ": "Kannur",
    "കാസറഗോഡ്": "Kasaragod",
}

ALERT_KEYWORDS = {
    "red": ["റെഡ്", "red alert", "Red"],
    "orange": ["ഓറഞ്ച്", "orange alert", "Orange"],
    "yellow": ["മഞ്ഞ", "yellow alert", "Yellow"],
}

VALID_LEVELS = ("red", "orange", "yellow")
ALL_DISTRICTS_EN = sorted(DISTRICT_MAP.values())


# ---------------------------------------------------------------------------
# 1. Fetch page
# ---------------------------------------------------------------------------

def fetch_page(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; mazha-live-bot/1.0; +https://mazha.live)"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.text


def rejoin_broken_number_lines(text: str) -> str:
    """Fixes a real observed KSDMA page artifact where a date/number gets
    split across lines mid-character (e.g. '30/07/2026' becomes a line
    containing just '3' followed by a line '0/07/2026: districts...').
    Any line that is PURELY 1-2 digits gets merged directly (no separator)
    with the next line, since that's never legitimate standalone content
    in this bulletin - it's always a broken number fragment."""
    lines = text.split("\n")
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if re.fullmatch(r"\d{1,2}", line) and i + 1 < len(lines):
            merged.append(line + lines[i + 1].strip())
            i += 2
        else:
            merged.append(lines[i])
            i += 1
    return "\n".join(merged)


def extract_main_text(html: str) -> str:
    """Pull visible text content from the page body. KSDMA is plain WordPress
    HTML, so grabbing all paragraph/text content and letting the LLM (or
    regex fallback) find the relevant bits works reliably without depending
    on a specific CSS class that may change.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return rejoin_broken_number_lines("\n".join(lines))


# ---------------------------------------------------------------------------
# 2. LLM extraction (Gemini primary, Groq fallback)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are extracting structured rainfall alert data from a Malayalam
government disaster-management bulletin (KSDMA, Kerala).

Return ONLY valid JSON — no markdown fences, no explanation, no extra text.

Schema:
{{
  "red": [ {{"date": "YYYY-MM-DD", "districts_ml": ["..."]}} ],
  "orange": [ {{"date": "YYYY-MM-DD", "districts_ml": ["..."]}} ],
  "yellow": [ {{"date": "YYYY-MM-DD", "districts_ml": ["..."]}} ]
}}

Rules:
- Only include entries that are explicitly about rainfall alerts (Red/Orange/Yellow),
  ignore unrelated page content (menus, footers, links).
- date must be in DD/MM/YYYY as written in source, converted to YYYY-MM-DD.
- districts_ml must be the exact Malayalam district names as written in the source text.
- If a section (red/orange/yellow) has no entries, return an empty list for it.
- Do not invent data. If unsure, omit the entry.

Source text:
---
{text}
---
"""


def _call_openai_compatible(url: str, api_key: str, model: str, prompt: str, provider_name: str) -> dict | None:
    """Shared caller — both Gemini and Groq expose an OpenAI-compatible
    /chat/completions endpoint, so one function handles both."""
    if not api_key:
        print(f"[info] {provider_name} API key not set, skipping.")
        return None

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        content = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
        data = json.loads(content)
        for level in VALID_LEVELS:
            data.setdefault(level, [])
        return data
    except Exception as e:
        print(f"[warn] {provider_name} extraction failed: {e}")
        return None


def extract_with_gemini(raw_text: str) -> dict | None:
    prompt = EXTRACTION_PROMPT.format(text=raw_text[:8000])
    return _call_openai_compatible(GEMINI_URL, GEMINI_API_KEY, GEMINI_MODEL, prompt, "Gemini")


def extract_with_groq(raw_text: str) -> dict | None:
    prompt = EXTRACTION_PROMPT.format(text=raw_text[:8000])
    return _call_openai_compatible(GROQ_URL, GROQ_API_KEY, GROQ_MODEL, prompt, "Groq")


# ---------------------------------------------------------------------------
# 3. Regex fallback extraction
# ---------------------------------------------------------------------------

def extract_with_regex(raw_text: str) -> dict:
    """Line-by-line scan: track the most recently seen alert-level keyword,
    then attach any date:districts entry to that level. Handles all real
    page structures seen in production:
      1. "DATE : districts" combined on one line
      2. Header text and the first date entry merged onto the SAME line
         (e.g. "ഓറഞ്ച് അലർട്ട് 30/07/2026: districts...") - a level-keyword
         match does NOT skip the rest of that line's processing.
      3. "DATE :" alone, with districts wrapping to the next line
    (Broken mid-date-string lines like "3" + "0/07/2026: ..." are already
    fixed upstream by rejoin_broken_number_lines() in extract_main_text.)
    """
    result = {level: [] for level in VALID_LEVELS}
    current_level = "yellow"  # KSDMA defaults to yellow context if no explicit marker yet

    lines = raw_text.split("\n")
    i = 0
    while i < len(lines):
        clean = lines[i].strip().strip("*").strip()
        if not clean:
            i += 1
            continue

        # Update current_level if this line mentions one - but DON'T skip
        # the rest of processing, since the date data might be on this
        # same line (header + first entry merged by the HTML extraction).
        for level, keywords in ALERT_KEYWORDS.items():
            if any(kw in clean for kw in keywords):
                current_level = level
                break

        # Case 1: "date : districts" combined, date at start of line
        m_combined = re.match(r"^(\d{2}/\d{2}/\d{4})\s*:\s*(.+)$", clean)
        if m_combined and m_combined.group(2).strip():
            date_str, district_blob = m_combined.groups()
            i += 1
        else:
            # Case 2: header text + date embedded later in the same line
            m_embedded = re.search(r"(\d{2}/\d{2}/\d{4})\s*:\s*(.+)$", clean)
            if m_embedded and m_embedded.group(2).strip():
                date_str, district_blob = m_embedded.groups()
                i += 1
            else:
                # Case 3: "date :" alone, districts on the NEXT non-empty line
                m_date_only = re.match(r"^(\d{2}/\d{2}/\d{4})\s*:?\s*$", clean)
                if m_date_only:
                    date_str = m_date_only.group(1)
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    district_blob = lines[j].strip() if j < len(lines) else ""
                    i = j + 1
                else:
                    i += 1
                    continue

        try:
            date_obj = datetime.strptime(date_str, "%d/%m/%Y")
        except ValueError:
            continue

        districts_ml = [d.strip() for d in re.split(r"[,،]", district_blob) if d.strip()]
        districts_ml = [d for d in districts_ml if d in DISTRICT_MAP]
        if districts_ml:
            result[current_level].append({
                "date": date_obj.strftime("%Y-%m-%d"),
                "districts_ml": districts_ml,
            })

    return result


# ---------------------------------------------------------------------------
# 4. Validate + enrich with English district names
# ---------------------------------------------------------------------------

def enrich_and_validate(data: dict) -> dict:
    clean = {level: [] for level in VALID_LEVELS}
    for level in VALID_LEVELS:
        for entry in data.get(level, []):
            date_str = entry.get("date", "")
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                continue  # skip invalid dates rather than guess

            districts_ml = entry.get("districts_ml", [])
            districts_en = []
            unmatched = []
            for d in districts_ml:
                d_stripped = d.strip()
                if d_stripped in DISTRICT_MAP:
                    districts_en.append(DISTRICT_MAP[d_stripped])
                else:
                    unmatched.append(d_stripped)

            if not districts_en:
                continue  # nothing usable in this entry

            clean_entry = {
                "date": date_str,
                "districts_ml": [d for d in districts_ml if d.strip() in DISTRICT_MAP],
                "districts_en": districts_en,
            }
            if unmatched:
                clean_entry["unmatched_districts"] = unmatched  # flagged, not dropped silently
            clean[level].append(clean_entry)
    return clean


def build_district_alert_map(data: dict) -> dict:
    district_map = {}
    for level in VALID_LEVELS:
        for entry in data.get(level, []):
            for d in entry["districts_en"]:
                district_map.setdefault(d, set()).add(level)
    return {k: sorted(v, key=lambda l: VALID_LEVELS.index(l)) for k, v in district_map.items()}


def get_available_dates(data: dict) -> list:
    """Returns the sorted list of dates this bulletin actually has data for.
    If today's date isn't in this list, "green" from build_district_colors
    means "no bulletin published for today yet", not "confirmed clear" -
    check this before assuming a fully-green map means a clear day."""
    dates = set()
    for level in VALID_LEVELS:
        for entry in data.get(level, []):
            dates.add(entry["date"])
    return sorted(dates)


def build_district_colors(data: dict, target_date: str | None = None) -> dict:
    """Simple flat map for coloring mazha.live's district map:
    { "Kozhikode": "red", "Wayanad": "red", "Ernakulam": "yellow", ...,
      "Kollam": "green" }  <- green = no active alert
    Every one of the 14 Kerala districts is always included.

    IMPORTANT: `data` can contain entries for several different dates at
    once (a 5-day forecast). Blending all of those together would show a
    district as an alert color today even if that alert is only valid for
    a future day. So this only colors a district based on alerts whose
    `date` matches `target_date` (defaults to today, IST).
    """
    if target_date is None:
        target_date = datetime.now(IST).strftime("%Y-%m-%d")

    todays_data = {
        level: [entry for entry in data.get(level, []) if entry.get("date") == target_date]
        for level in VALID_LEVELS
    }
    district_map = build_district_alert_map(todays_data)
    colors = {}
    for district in ALL_DISTRICTS_EN:
        levels = district_map.get(district)
        colors[district] = levels[0] if levels else "green"  # levels[0] = highest severity (red > orange > yellow)
    return colors


# ---------------------------------------------------------------------------
# 5. Diff against last saved state
# ---------------------------------------------------------------------------

def load_previous_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"alerts": {level: [] for level in VALID_LEVELS}}


def entry_key(level, entry):
    return (level, entry["date"], tuple(sorted(entry["districts_en"])))


def find_new_alerts(previous: dict, current: dict) -> list:
    prev_keys = set()
    for level in VALID_LEVELS:
        for entry in previous.get("alerts", {}).get(level, []):
            prev_keys.add(entry_key(level, entry))

    new_alerts = []
    for level in VALID_LEVELS:
        for entry in current.get(level, []):
            key = entry_key(level, entry)
            if key not in prev_keys:
                new_alerts.append({"level": level, **entry})
    return new_alerts


def save_state(current: dict):
    colors = build_district_colors(current)
    available_dates = get_available_dates(current)
    scraped_at = datetime.now(IST).isoformat()
    today_str = datetime.now(IST).strftime("%Y-%m-%d")

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "alerts": current,
            "district_alert_map": build_district_alert_map(current),
            "district_colors": colors,
            "available_dates": available_dates,
            "scraped_at": scraped_at,
        }, f, ensure_ascii=False, indent=2)

    # Separate, minimal file — this is the one your map should fetch.
    with open(COLORS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "scraped_at": scraped_at,
            "date": today_str,
            "has_data_for_today": today_str in available_dates,
            "available_dates": available_dates,
            "colors": colors,
        }, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(debug: bool = False):
    print(f"[info] Fetching {KSDMA_URL} ...")
    html = fetch_page(KSDMA_URL)
    raw_text = extract_main_text(html)

    if debug:
        print("\n" + "=" * 60)
        print("DEBUG: FULL raw_text (every line, numbered)")
        print("=" * 60)
        for idx, line in enumerate(raw_text.split("\n")):
            print(f"{idx:3d}: {repr(line)}")

    extracted = extract_with_gemini(raw_text)
    used_method = "gemini"

    if extracted is None:
        print("[info] Gemini unavailable, trying Groq...")
        extracted = extract_with_groq(raw_text)
        used_method = "groq"

    if extracted is None:
        print("[info] Both LLMs unavailable, falling back to regex extraction.")
        extracted = extract_with_regex(raw_text)
        used_method = "regex"

    if debug:
        print("\n" + "=" * 60)
        print(f"DEBUG: extraction method used = {used_method}")
        print("DEBUG: extracted (before enrichment)")
        print("=" * 60)
        print(json.dumps(extracted, ensure_ascii=False, indent=2))

    validated = enrich_and_validate(extracted)
    colors = build_district_colors(validated)

    previous = load_previous_state()
    new_alerts = find_new_alerts(previous, validated)

    print("\n=== District colors (for map) ===")
    print(json.dumps(colors, ensure_ascii=False, indent=2))

    if new_alerts:
        print(f"\n[ALERT] {len(new_alerts)} new alert(s) detected!")
        for a in new_alerts:
            districts = ", ".join(a["districts_en"])
            print(f"  -> {a['level'].upper()} | {a['date']} | {districts}")
    else:
        print("\n[info] No new alerts since last run.")

    save_state(validated)
    return {"colors": colors, "new_alerts": new_alerts}


if __name__ == "__main__":
    debug_mode = "--debug" in sys.argv
    try:
        run(debug=debug_mode)
    except requests.RequestException as e:
        print(f"[error] Failed to fetch KSDMA page: {e}", file=sys.stderr)
        sys.exit(1)
