import re
from typing import Dict


INDIAN_PLATE_REGEX = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$")
VALID_STATE_CODES = {
    "AN",
    "AP",
    "AR",
    "AS",
    "BR",
    "CG",
    "CH",
    "DD",
    "DL",
    "DN",
    "GA",
    "GJ",
    "HR",
    "HP",
    "JH",
    "JK",
    "KA",
    "KL",
    "LA",
    "LD",
    "MH",
    "ML",
    "MN",
    "MP",
    "MZ",
    "NL",
    "OD",
    "PB",
    "PY",
    "RJ",
    "SK",
    "TN",
    "TR",
    "TS",
    "UK",
    "UP",
    "WB",
}


def normalize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def validate_plate_format(text: str, country: str = "india") -> Dict[str, str]:
    """
    Validate the OCR result against the selected country's plate format.
    The project currently ships with Indian private/commercial plate rules.
    """
    normalized = normalize_plate_text(text)
    result = {
        "country": country.lower(),
        "normalized_text": normalized,
        "status": "INVALID",
        "reason": "No plate text was extracted.",
    }

    if not normalized:
        return result

    if country.lower() != "india":
        result["status"] = "SUSPICIOUS"
        result["reason"] = "Unsupported country profile. Only Indian format is configured."
        return result

    if len(normalized) < 4:
        result["reason"] = "Plate is too short to be a valid vehicle registration."
        return result

    if len(normalized) < 8:
        result["status"] = "SUSPICIOUS"
        result["reason"] = "Plate is shorter than the expected Indian registration format."
        return result

    if len(normalized) > 10:
        result["status"] = "SUSPICIOUS"
        result["reason"] = "Plate is longer than the expected Indian registration format."
        return result

    if not normalized[:2].isalpha():
        result["status"] = "SUSPICIOUS"
        result["reason"] = "First two characters should be a state or union territory code."
        return result

    if normalized[:2] not in VALID_STATE_CODES:
        result["status"] = "SUSPICIOUS"
        result["reason"] = "Unknown state code for an Indian registration number."
        return result

    if len(normalized) >= 4 and not normalized[2:4].isdigit():
        result["status"] = "SUSPICIOUS"
        result["reason"] = "The district code should be two digits."
        return result

    if not normalized[-4:].isdigit():
        result["status"] = "SUSPICIOUS"
        result["reason"] = "The last four characters should be digits."
        return result

    middle_series = normalized[4:-4]
    if not middle_series:
        result["reason"] = "Missing alphabetic series block before the final four digits."
        return result

    if not middle_series.isalpha():
        result["status"] = "SUSPICIOUS"
        result["reason"] = "The series block contains non-alphabetic characters."
        return result

    if len(middle_series) > 3:
        result["status"] = "SUSPICIOUS"
        result["reason"] = "Series block is longer than the common 1 to 3 letter range."
        return result

    if INDIAN_PLATE_REGEX.fullmatch(normalized):
        result["status"] = "VALID"
        result["reason"] = "Matches the standard Indian vehicle registration pattern."
        return result

    result["status"] = "SUSPICIOUS"
    result["reason"] = "Close to a real Indian format, but not a clean full regex match."
    return result
