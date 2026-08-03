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
  5. ALSO run a dedicated regex pass for the IMD 5-day forecast block
     (a differently-worded paragraph KSDMA embeds alongside its own
     alert listing) and merge it in, however extraction happened above
  6. Validate districts against known Kerala district list
  7. Diff against last saved state -> report new alerts
  8. Save current state for next run

Usage:
  export GEMINI_API_KEY="your_key_here"   # primary  - aistudio.google.com
  export GROQ_API_KEY="your_key_here"     # fallback - console.groq.com/keys
  python ksdma_scraper.py

Run this on a schedule (cron / GitHub Actions / cloud function) every 1-2 hours.

---------------------------------------------------------------------------
CHANGELOG (Aug 2026 bugfix)
---------------------------------------------------------------------------
KSDMA started publishing lines with TWO dates joined by '&' before a single
shared district list, e.g.:

    03/08/2026 & 04/08/2026: പത്തനംതിട്ട, ആലപ്പുഴ, കോട്ടയം, ...
    03/08/2026 & 04/08/2026: തിരുവനന്തപുരം, കൊല്ലം

The old single-date regex (`(\\d{2}/\\d{2}/\\d{4})\\s*:?\\s*(.*)`) only
captured the FIRST date; everything after it -- including the literal text
"& 04/08/2026:" -- was swallowed into the district blob. Since that leading
"& DD/MM/YYYY:" fragment doesn't exactly match any entry in DISTRICT_MAP,
whichever district happened to be listed FIRST after it got silently
dropped (it looked like an unmatched/junk token), and the second date was
never registered as its own alert entry at all.

Symptoms this caused:
  - Thiruvananthapuram dropped from the Yellow entry (always listed first
    in that line).
  - Pathanamthitta dropped from the Orange entry (always listed first in
    that line).
  - Kollam kept showing but only for the first of its two alert dates,
    so it could flip back to green a day early.

Fix: `parse_date_district_line()` now extracts ALL date tokens on a line
and treats the text after the LAST date token as the shared district list,
emitting one entry per date. Both `extract_with_regex()` and
`extract_imd_forecast_block()` now use this shared helper so the fix
applies everywhere a date/district line is parsed.
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
    "തൃശ്ശൂർ": "Thrissur",  # alternate spelling KSDMA itself uses inconsistently (extra ശ)
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

# Maps bare Malayalam color words -> level, used by both the generic
# level-declaration check and the IMD-block level heading check.
IMD_COLOR_ML_MAP = {
    "റെഡ്": "red",
    "ഓറഞ്ച്": "orange",
    "മഞ്ഞ": "yellow",
}


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
    # Collapse excess blank lines
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Groq LLM extraction
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
- This includes both KSDMA's own alert listing AND any IMD (Indian
  Meteorological Department / "കേന്ദ്ര കാലാവസ്ഥ വകുപ്പ്") 5-day forecast
  alert block on the page — extract alerts from both sources into the
  same schema.
- Some lines list TWO dates joined by '&' before a single shared district
  list (e.g. "03/08/2026 & 04/08/2026: <districts>"). In that case, emit
  ONE separate entry per date, each with the same districts_ml list.
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
        # Strip accidental markdown fences some models add despite instructions
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

def _is_level_declaration_line(clean: str) -> bool:
    return any(kw in clean for keywords in ALERT_KEYWORDS.values() for kw in keywords)


# Matches a single DD/MM/YYYY date token anywhere in a line. Used (instead
# of anchoring only at the start of the line) so we can find every date
# token on lines that list more than one, e.g. "DD/MM/YYYY & DD/MM/YYYY:".
DATE_TOKEN_RE = re.compile(r"\d{2}/\d{2}/\d{4}")


def parse_date_district_line(clean: str):
    """Parse a line that may contain one or more DD/MM/YYYY date tokens
    followed by a shared, comma-separated district list, e.g.:

        "24/07/2026 : മലപ്പുറം, കോഴിക്കോട്"                     (single date)
        "03/08/2026 & 04/08/2026: പത്തനംതിട്ട, ആലപ്പുഴ, ..."   (multi-date)

    Returns (list_of_date_strings, district_blob_text) if at least one date
    token is found, or (None, None) if the line has no date token at all
    (i.e. it's not a date/district line, e.g. it's a heading or prose).

    Only text AFTER THE LAST date token is treated as the district blob —
    this is what fixes the previous bug where, for multi-date lines, the
    second "& DD/MM/YYYY:" fragment was incorrectly swallowed into the
    district list, silently dropping whichever district was listed first
    (since "& DD/MM/YYYY: <first district>" never exactly matches a
    DISTRICT_MAP entry) and losing the second date as an entry entirely.
    """
    matches = list(DATE_TOKEN_RE.finditer(clean))
    if not matches:
        return None, None
    dates = [m.group(0) for m in matches]
    tail = clean[matches[-1].end():]
    tail = tail.lstrip(" :&,،")  # strip leading ':' / '&' / whitespace / commas
    return dates, tail.strip()


def extract_with_regex(raw_text: str) -> dict:
    """Line-by-line scan: track the most recently seen alert-level keyword,
    then attach any date:districts line found after it to that level.

    Handles both single-date lines and lines with two dates joined by '&'
    sharing one district list (see parse_date_district_line docstring),
    as well as the case where the source splits "DD/MM/YYYY :" and its
    district list across two separate lines."""
    result = {level: [] for level in VALID_LEVELS}
    current_level = "yellow"  # KSDMA defaults to yellow context if no explicit marker yet

    lines = raw_text.split("\n")
    i = 0
    while i < len(lines):
        clean = lines[i].strip().strip("*").strip()
        i += 1
        if not clean:
            continue

        # Check if this line declares an alert level
        matched_level = None
        for level, keywords in ALERT_KEYWORDS.items():
            if any(kw in clean for kw in keywords):
                matched_level = level
                break
        if matched_level:
            current_level = matched_level
            continue

        # Check if this line is a date:districts entry (possibly multi-date)
        dates, district_blob = parse_date_district_line(clean)
        if dates is None:
            continue

        # If the date line has nothing after it (e.g. "24/07/2026 :" on its
        # own), the district list is likely on the next line.
        if not district_blob and i < len(lines):
            next_line = lines[i].strip()
            if next_line and not DATE_TOKEN_RE.search(next_line) and not _is_level_declaration_line(next_line):
                district_blob = next_line
                i += 1

        districts_ml = [d.strip() for d in re.split(r"[,،]", district_blob) if d.strip()]
        # Filter to only known districts (avoids picking up unrelated text)
        districts_ml = [d for d in districts_ml if d in DISTRICT_MAP]
        if not districts_ml:
            continue

        for date_str in dates:
            try:
                date_obj = datetime.strptime(date_str, "%d/%m/%Y")
            except ValueError:
                continue
            result[current_level].append({
                "date": date_obj.strftime("%Y-%m-%d"),
                "districts_ml": districts_ml,
            })

    return result


# ---------------------------------------------------------------------------
# 3b. IMD 5-day forecast block extraction (additive, additional regex pass)
# ---------------------------------------------------------------------------
#
# KSDMA also embeds a separate "IMD / കേന്ദ്ര കാലാവസ്ഥ വകുപ്പ്" 5-day
# rainfall-forecast block on the page, worded differently from its own
# alert listing. This wording has changed over time:
#
#   OLDER format — the level is named inline, in the same sentence:
#     "...കാലാവസ്ഥ വകുപ്പ് മഞ്ഞ (Yellow) അലർട്ട് പ്രഖ്യാപിച്ചിരിക്കുന്നു."
#     24/07/2026 : മലപ്പുറം, കോഴിക്കോട്, വയനാട്, കണ്ണൂർ, കാസറഗോഡ്
#     25/07/2026 : കണ്ണൂർ, കാസറഗോഡ്
#
#   CURRENT format (Aug 2026) — a single intro sentence just mentions
#   "കേന്ദ്ര കാലാവസ്ഥ വകുപ്പ്" once ("...ഓറഞ്ച്, മഞ്ഞ അലർട്ടുകൾ
#   പ്രഖ്യാപിച്ചിരിക്കുന്നു."), then each level gets its own short plain
#   Malayalam heading ("ഓറഞ്ച് അലർട്ട്", "മഞ്ഞ അലർട്ട്") with NO English
#   color word and NOT on the same line as "കേന്ദ്ര കാലാവസ്ഥ വകുപ്പ്" at
#   all, followed by that level's (possibly multi-date) date/district
#   lines.
#
# To handle both without over-fitting to either, this function:
#   1. Confirms an IMD section exists by finding the first line mentioning
#      "കേന്ദ്ര കാലാവസ്ഥ വകുപ്പ്".
#   2. From there, scans forward for ANY level-heading line (English
#      color-in-parens OR a bare Malayalam color word next to "അലർട്ട്"),
#      sets that as the current level, and reads forward through
#      consecutive date/district lines using the same
#      parse_date_district_line() helper the main regex path uses (so
#      multi-date lines are handled identically here).
#   3. Switches level whenever a new level-heading line appears, and clears
#      the current level whenever a non-heading, non-date (i.e. prose) line
#      is seen, so stray digits in descriptive paragraphs are never
#      misattributed to a level.
#
# Note: on the CURRENT page format, the generic extract_with_regex() pass
# already picks up "ഓറഞ്ച് അലർട്ട്" / "മഞ്ഞ അലർട്ട്" headings fine on its
# own (they match ALERT_KEYWORDS directly) — this function is mainly a
# safety net for the older inline-sentence format. Any overlap between the
# two passes is de-duplicated by merge_extraction_results().

IMD_SECTION_MARKER_RE = re.compile(r"കേന്ദ്ര\s*കാലാവസ്ഥ\s*വകുപ്പ്")
IMD_LEVEL_LINE_RE = re.compile(
    r"(?:\((Red|Orange|Yellow)\)|(റെഡ്|ഓറഞ്ച്|മഞ്ഞ))\s*അലർട്ട്",
    re.IGNORECASE,
)


def extract_imd_forecast_block(raw_text: str) -> dict:
    result = {level: [] for level in VALID_LEVELS}
    lines = [ln.strip() for ln in raw_text.split("\n")]

    imd_start = None
    for idx, line in enumerate(lines):
        if IMD_SECTION_MARKER_RE.search(line):
            imd_start = idx
            break
    if imd_start is None:
        return result  # no IMD section on this page at all

    current_level = None
    i = imd_start
    while i < len(lines):
        line = lines[i].strip("*").strip()
        if not line:
            i += 1
            continue

        level_match = IMD_LEVEL_LINE_RE.search(line)
        if level_match:
            color_en, color_ml = level_match.group(1), level_match.group(2)
            current_level = color_en.lower() if color_en else IMD_COLOR_ML_MAP[color_ml]
            i += 1
            continue

        dates, district_blob = parse_date_district_line(line)
        if dates is not None and current_level is not None:
            if not district_blob and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and not DATE_TOKEN_RE.search(next_line) and not _is_level_declaration_line(next_line):
                    district_blob = next_line
                    i += 1

            districts_ml = [d.strip() for d in re.split(r"[,،]", district_blob) if d.strip()]
            districts_ml = [d for d in districts_ml if d in DISTRICT_MAP]
            if districts_ml:
                for date_str in dates:
                    try:
                        date_obj = datetime.strptime(date_str, "%d/%m/%Y")
                    except ValueError:
                        continue
                    result[current_level].append({
                        "date": date_obj.strftime("%Y-%m-%d"),
                        "districts_ml": districts_ml,
                    })
            i += 1
            continue

        # Neither a level heading nor a date line -> we're in descriptive
        # prose. Clear the current level so stray digits/text in paragraphs
        # between sections are never misattributed to the previous level.
        # Scanning continues (rather than stopping) because the page
        # typically has more level sections further down.
        current_level = None
        i += 1

    return result


def merge_extraction_results(*results: dict) -> dict:
    """Merge multiple {level: [entries]} dicts (e.g. LLM/regex output plus
    the IMD-forecast pass), de-duplicating identical (level, date,
    districts_ml) entries so the same alert isn't double-counted if more
    than one pass happens to pick it up."""
    merged = {level: [] for level in VALID_LEVELS}
    seen = set()
    for res in results:
        if not res:
            continue
        for level in VALID_LEVELS:
            for entry in res.get(level, []):
                key = (level, entry.get("date"), tuple(sorted(entry.get("districts_ml", []))))
                if key in seen:
                    continue
                seen.add(key)
                merged[level].append(entry)
    return merged


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


ALL_DISTRICTS_EN = sorted(DISTRICT_MAP.values())


def build_district_colors(data: dict, target_date: str | None = None) -> dict:
    """Simple flat map for coloring mazha.live's district map:
    { "Kozhikode": "red", "Wayanad": "red", "Ernakulam": "yellow", ...,
      "Kollam": "green" }  <- green = no active alert
    Every one of the 14 Kerala districts is always included, so the map
    always has a color to render, even on a clear day.

    IMPORTANT: `data` (e.g. from enrich_and_validate) can contain entries
    for several different dates at once — the IMD block alone gives a
    5-day forecast. Blending all of those together would show a district
    as "yellow" today even if its alert is only for two days from now.
    So this only colors a district based on alerts whose `date` matches
    `target_date`. The full multi-day picture is still available via
    build_district_alert_map(data) / the saved "alerts" list, this
    function just answers "what's active right now".

    target_date default (when not explicitly passed):
      1. If the wall-clock date (IST) has an entry in the scraped data,
         use it — this is the normal case, showing today's actual alert.
      2. Else, if the wall-clock date is BEFORE every date in the data
         (the forecast was just published slightly ahead of schedule, or
         the local clock is a little behind), use the EARLIEST upcoming
         date present — i.e. "day 1" of the forecast.
      3. Else (the wall-clock date is AFTER every date in the data — the
         whole forecast has expired/gone stale), use the wall-clock date
         as-is. It won't match anything, so every district correctly
         comes back "green" rather than showing a leftover alert from a
         forecast that's no longer current.
    This avoids two failure modes seen before: always using the earliest
    published date (which goes stale once that day passes) and always
    trusting the raw wall clock (which can drift a day out of sync with
    what KSDMA has actually published).
    """
    if target_date is None:
        wall_clock_today = datetime.now(IST).strftime("%Y-%m-%d")
        all_dates = sorted({
            entry.get("date")
            for level in VALID_LEVELS
            for entry in data.get(level, [])
            if entry.get("date")
        })
        if wall_clock_today in all_dates:
            target_date = wall_clock_today
        else:
            upcoming = [d for d in all_dates if d >= wall_clock_today]
            target_date = upcoming[0] if upcoming else wall_clock_today

    todays_data = {
        level: [entry for entry in data.get(level, []) if entry.get("date") == target_date]
        for level in VALID_LEVELS
    }
    district_map = build_district_alert_map(todays_data)  # e.g. {"Kozhikode": ["red", "yellow"]}
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


COLORS_FILE = os.path.join(os.path.dirname(__file__), "district_colors.json")


def save_state(current: dict):
    colors = build_district_colors(current)
    scraped_at = datetime.now(IST).isoformat()

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "alerts": current,
            "district_alert_map": build_district_alert_map(current),
            "district_colors": colors,
            "scraped_at": scraped_at,
        }, f, ensure_ascii=False, indent=2)

    # Separate, minimal file — this is the one your map should fetch.
    with open(COLORS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "scraped_at": scraped_at,
            "colors": colors,
        }, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print(f"[info] Fetching {KSDMA_URL} ...")
    html = fetch_page(KSDMA_URL)
    raw_text = extract_main_text(html)

    print("\n=== DEBUG: raw_text length ===")
    print(len(raw_text))
    print("\n=== DEBUG: raw_text (first 3000 chars) ===")
    print(raw_text[:3000])

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

    # Safety net: ALWAYS also run the deterministic regex extractor and
    # merge it in, even when Gemini/Groq already produced a result. LLM
    # extraction can silently miss/misclassify an entry (e.g. drop a
    # district from a multi-section Orange+Yellow bulletin) with no error
    # to catch it. The regex pass is deterministic and cheap, so running
    # it unconditionally and merging (de-duped) is strictly additive —
    # it only adds entries the primary method missed, never removes any.
    if used_method != "regex":
        regex_extra = extract_with_regex(raw_text)
        extracted = merge_extraction_results(extracted, regex_extra)

    # Additive pass: pick up the IMD 5-day forecast block regardless of
    # which method above ran, then merge (with de-dup) into `extracted`.
    imd_extra = extract_imd_forecast_block(raw_text)
    if any(imd_extra[level] for level in VALID_LEVELS):
        print(f"[info] IMD forecast block found ({used_method} was primary method), merging in.")
    extracted = merge_extraction_results(extracted, imd_extra)

    print("\n=== DEBUG: extracted (before enrichment) ===")
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
    try:
        run()
    except requests.RequestException as e:
        print(f"[error] Failed to fetch KSDMA page: {e}", file=sys.stderr)
        sys.exit(1)
