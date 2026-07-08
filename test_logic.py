import json
import sys
sys.path.insert(0, ".")
from ksdma_scraper import (
    extract_with_regex, enrich_and_validate, build_district_alert_map,
    find_new_alerts, build_district_colors, ALL_DISTRICTS_EN
)

sample_text = """
കേരള സംസ്ഥാന ദുരന്ത നിവാരണ അതോറിറ്റി
വിവിധ ജില്ലകളിൽ കേന്ദ്ര കാലാവസ്ഥ വകുപ്പ് മഞ്ഞ (Yellow) അലർട്ട് പ്രഖ്യാപിച്ചിരിക്കുന്നു
21/06/2026 : പത്തനംതിട്ട, ആലപ്പുഴ, കോട്ടയം, ഇടുക്കി, എറണാകുളം, തൃശൂർ, മലപ്പുറം, കോഴിക്കോട്
22/06/2026 : ആലപ്പുഴ, എറണാകുളം, തൃശൂർ, മലപ്പുറം, കോഴിക്കോട്, വയനാട്, കണ്ണൂർ, കാസറഗോഡ്
23/06/2026 : കണ്ണൂർ, കാസറഗോഡ്

കേരളത്തിൽ അതിതീവ്ര മഴയ്ക്ക് സാധ്യതയുള്ളതിനാൽ കേന്ദ്ര കാലാവസ്ഥ വകുപ്പ് റെഡ് അലർട്ട് പ്രഖ്യാപിച്ചിരിക്കുന്നു.
റെഡ് അലർട്ട്
07/07/2026: കോഴിക്കോട്, വയനാട്

പുറപ്പെടുവിച്ച സമയം: 01.00 PM; 07/07/2026
"""

print("=== Testing regex extraction on mixed-format sample (both formats seen in conversation) ===\n")

extracted = extract_with_regex(sample_text)
print("Raw extracted (before enrichment):")
print(json.dumps(extracted, ensure_ascii=False, indent=2))

validated = enrich_and_validate(extracted)
print("\nValidated + enriched (English names added):")
print(json.dumps(validated, ensure_ascii=False, indent=2))

district_map = build_district_alert_map(validated)
print("\nDistrict alert map (severity-sorted):")
print(json.dumps(district_map, ensure_ascii=False, indent=2))

# Test diffing logic: simulate "previous state" only had the yellow alerts
previous_state = {
    "alerts": {
        "red": [],
        "orange": [],
        "yellow": validated["yellow"],  # same yellow entries as "already seen"
    }
}
new_alerts = find_new_alerts(previous_state, validated)
print("\nNew alerts detected (should be the Red alert only, since yellow was already known):")
print(json.dumps(new_alerts, ensure_ascii=False, indent=2))

assert len(new_alerts) == 1, f"Expected 1 new alert, got {len(new_alerts)}"
assert new_alerts[0]["level"] == "red", "Expected the new alert to be 'red'"
assert "Kozhikode" in new_alerts[0]["districts_en"]
assert "Wayanad" in new_alerts[0]["districts_en"]
print("\n✅ ALL ASSERTIONS PASSED — diffing correctly isolates the new Red alert.")

# --- Test the simple district-color map (for map coloring) ---
colors = build_district_colors(validated)
print("\nDistrict colors (simple map for mazha.live):")
print(json.dumps(colors, ensure_ascii=False, indent=2))

assert len(colors) == 14, f"Expected all 14 districts, got {len(colors)}"
assert colors["Kozhikode"] == "red", "Kozhikode has a Red alert, should show red (highest severity)"
assert colors["Wayanad"] == "red", "Wayanad has a Red alert, should show red"
assert colors["Ernakulam"] == "yellow", "Ernakulam only has Yellow, should show yellow"
assert colors["Kollam"] == "green", "Kollam has no alerts at all, should default to green"
assert colors["Thiruvananthapuram"] == "green", "Thiruvananthapuram has no alerts, should default to green"
print("\n✅ ALL COLOR-MAP ASSERTIONS PASSED — every district covered, red beats yellow, no-alert = green.")
