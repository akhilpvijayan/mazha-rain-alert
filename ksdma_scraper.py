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

def extract_with_regex(raw_text: str) -> dict:
    """Line-by-line scan: track the most recently seen alert-level keyword,
    then attach any date:districts line found after it to that level."""
    result = {level: [] for level in VALID_LEVELS}
    current_level = "yellow"  # KSDMA defaults to yellow context if no explicit marker yet

    date_line_re = re.compile(r"(\d{2}/\d{2}/\d{4})\s*:?\s*(.+)")

    for line in raw_text.split("\n"):
        clean = line.strip().strip("*").strip()
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

        # Check if this line is a date:districts entry
        m = date_line_re.match(clean)
        if m:
            date_str, district_blob = m.groups()
            try:
                date_obj = datetime.strptime(date_str, "%d/%m/%Y")
            except ValueError:
                continue
            districts_ml = [d.strip() for d in re.split(r"[,،]", district_blob) if d.strip()]
            # Filter to only known districts (avoids picking up unrelated text)
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


ALL_DISTRICTS_EN = sorted(DISTRICT_MAP.values())


def build_district_colors(data: dict, target_date: str | None = None) -> dict:
    """Simple flat map for coloring mazha.live's district map, filtered to a
    SPECIFIC date (defaults to today, IST) so a downgrade/upgrade forecast
    for a future day doesn't get conflated with today's actual alert level.

    e.g. if Kozhikode is Red on 08/07 but only Yellow on 09/07, calling this
    with target_date="2026-07-08" correctly returns "red" for Kozhikode,
    and calling it with "2026-07-09" correctly returns "yellow" instead.

    Every one of the 14 Kerala districts is always included, defaulting to
    "green" if no alert is listed for that district on that date.
    """
    if target_date is None:
        target_date = datetime.now(IST).strftime("%Y-%m-%d")

    # Build per-district severity set, but only from entries matching target_date
    district_levels = {}
    for level in VALID_LEVELS:
        for entry in data.get(level, []):
            if entry["date"] != target_date:
                continue
            for d in entry["districts_en"]:
                district_levels.setdefault(d, set()).add(level)

    colors = {}
    for district in ALL_DISTRICTS_EN:
        levels = district_levels.get(district)
        if levels:
            # pick highest severity among same-date entries (red > orange > yellow)
            highest = sorted(levels, key=lambda l: VALID_LEVELS.index(l))[0]
            colors[district] = highest
        else:
            colors[district] = "green"
    return colors


def build_forecast_colors(data: dict) -> dict:
    """Same as build_district_colors but returns a map of
    { date: { district: color } } for every date present in the bulletin —
    useful if you ever want a 'next few days' view instead of just today."""
    all_dates = set()
    for level in VALID_LEVELS:
        for entry in data.get(level, []):
            all_dates.add(entry["date"])

    return {d: build_district_colors(data, target_date=d) for d in sorted(all_dates)}


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
    colors = build_district_colors(current)  # today only
    forecast = build_forecast_colors(current)  # all dates in bulletin
    scraped_at = datetime.now(IST).isoformat()

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "alerts": current,
            "district_alert_map": build_district_alert_map(current),
            "district_colors": colors,
            "forecast_colors": forecast,
            "scraped_at": scraped_at,
        }, f, ensure_ascii=False, indent=2)

    # Separate, minimal file — this is the one your map should fetch.
    with open(COLORS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "scraped_at": scraped_at,
            "date": datetime.now(IST).strftime("%Y-%m-%d"),
            "colors": colors,
        }, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print(f"[info] Fetching {KSDMA_URL} ...")
    html = fetch_page(KSDMA_URL)
    raw_text = extract_main_text(html)

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

    validated = enrich_and_validate(extracted)
    colors = build_district_colors(validated)

    previous = load_previous_state()
    new_alerts = find_new_alerts(previous, validated)

    # ---- This is the simple output for coloring your map ----
    print("\n=== District colors (for map) ===")
    print(json.dumps(colors, ensure_ascii=False, indent=2))

    if new_alerts:
        print(f"\n[ALERT] {len(new_alerts)} new alert(s) detected!")
        for a in new_alerts:
            districts = ", ".join(a["districts_en"])
            print(f"  -> {a['level'].upper()} | {a['date']} | {districts}")
        try:
            from fcm_notifier import send_alert_notification
            send_alert_notification(new_alerts)
        except Exception as e:
            print(f"[warn] Could not send push notification: {e}")
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
