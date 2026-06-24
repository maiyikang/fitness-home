from __future__ import annotations

import re
from typing import Optional


def clean_text(value: Optional[str]) -> str:
    """
    Clean text values from CSV files.

    - Convert None to empty string.
    - Remove unusual line separators.
    - Replace multiple spaces.
    - Decode common HTML entities.
    """

    if value is None:
        return ""

    text = str(value)

    text = text.replace("&amp;", "&")
    text = text.replace("\u2028", " ")
    text = text.replace("\u2029", " ")
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_float(value: Optional[str], default: float = 0.0) -> float:
    """
    Safely convert text to float.
    """

    try:
        text = clean_text(value)

        if text == "":
            return default

        return float(text)

    except (TypeError, ValueError):
        return default


def parse_int(value: Optional[str], default: int = 0) -> int:
    """
    Safely convert text to integer.
    """

    try:
        text = clean_text(value)

        if text == "":
            return default

        return int(float(text))

    except (TypeError, ValueError):
        return default


def parse_price(value: Optional[str]) -> float:
    """
    Extract numeric price from strings.

    Example:

    "$12.99" -> 12.99

    "£18" -> 18

    "$$" -> 0
    """

    text = clean_text(value)

    if text == "":
        return 0.0

    match = re.search(r"\d+(\.\d+)?", text)

    if match is None:
        return 0.0

    return float(match.group())


def is_empty(value: Optional[str]) -> bool:
    """
    Check whether a field is empty after cleaning.
    """

    return clean_text(value) == ""


def normalize_category(category: Optional[str]) -> str:
    """
    Normalize restaurant categories.

    Example:
        'Sushi,  Japanese'
        ->
        'Sushi, Japanese'
    """

    text = clean_text(category)

    if text == "":
        return "Unknown"

    items = [item.strip() for item in text.split(",")]

    items = [item for item in items if item != ""]

    return ", ".join(items)