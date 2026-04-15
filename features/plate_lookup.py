import json
from pathlib import Path
from typing import Dict, List

from features.plate_validation import normalize_plate_text


DEFAULT_WATCHLIST_PATH = Path("features") / "data" / "vehicle_watchlist.json"


def load_vehicle_database(database_path: Path | str = DEFAULT_WATCHLIST_PATH) -> List[Dict]:
    path = Path(database_path)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    for record in records:
        record["plate_number"] = normalize_plate_text(record.get("plate_number", ""))
    return records


def lookup_plate_record(
    plate_text: str, database_path: Path | str = DEFAULT_WATCHLIST_PATH
) -> Dict:
    normalized = normalize_plate_text(plate_text)
    records = load_vehicle_database(database_path)

    for record in records:
        if record.get("plate_number") == normalized:
            reasons = []
            if record.get("stolen"):
                reasons.append("Reported stolen vehicle")
            if record.get("flagged_owner"):
                reasons.append("Owner flagged for manual verification")
            if record.get("registration_status") == "expired":
                reasons.append("Registration expired")
            if not reasons:
                reasons.append("Matched a watchlist record")

            return {
                "match_found": True,
                "alert_level": "HIGH" if record.get("stolen") else "MEDIUM",
                "plate_number": normalized,
                "reasons": reasons,
                "record": record,
            }

    return {
        "match_found": False,
        "alert_level": "NONE",
        "plate_number": normalized,
        "reasons": ["No watchlist match found in the local vehicle database."],
        "record": None,
    }
