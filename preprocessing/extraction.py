"""Date and event extraction for the academic calendar.

One keyword ladder, used by both the indexing path (text_splitter) and the
retrieval path (app.retrieval), so the two can no longer drift apart.
Pure functions, no I/O.
"""
import re
from datetime import datetime

TURKISH_MONTHS = {
    'Oca': 'January', 'Şub': 'February', 'Mar': 'March',
    'Nis': 'April', 'May': 'May', 'Haz': 'June',
    'Tem': 'July', 'Ağu': 'August', 'Eyl': 'September',
    'Eki': 'October', 'Kas': 'November', 'Ara': 'December'
}

# Ordered: first match wins. Same ladder for chunk text and user queries.
EVENT_KEYWORDS = [
    ("deadline", ["son gün", "deadline", "son tarih"]),
    ("period", ["arasında"]),
    ("holiday", ["tatil", "bayram"]),
    ("exam", ["sınav"]),
    ("registration", ["kayıt", "kaydı"]),
    ("graduation", ["mezuniyet"]),
    ("application", ["başvur"]),
    ("announcement", ["duyuru", "ilan"]),
    ("course", ["ders"]),
]


def parse_date(date_str):
    """Parse Turkish date format (e.g. '14.Şub.25 Cuma') into datetime."""
    try:
        # Remove day name if present
        date_str = re.sub(r'\s+[PÇCPSPÇC]\w+$', '', date_str)

        parts = date_str.split('.')
        if len(parts) != 3:
            return None

        day, month, year = parts
        month = TURKISH_MONTHS.get(month, month)
        year = '20' + year if int(year) < 50 else '19' + year

        return datetime.strptime(f"{day} {month} {year}", "%d %B %Y")
    except (ValueError, AttributeError):
        return None


def extract_event_type(text, default=None):
    """Return the event type for a chunk or a query.

    Indexing passes default="event" (every chunk gets a label);
    querying passes default=None (no match means no filter).
    """
    text = text.lower()
    for event_type, keywords in EVENT_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return event_type
    return default


def extract_academic_period(text):
    """Return 'fall' / 'spring' / 'summer' or None."""
    text = text.lower()
    if "güz" in text:
        return "fall"
    if "bahar" in text:
        return "spring"
    # Word boundary so 'yazılı' (written) doesn't match 'yaz' (summer)
    if re.search(r'\byaz\b', text):
        return "summer"
    return None


def is_date_line(line):
    """Check if a calendar line starts with a date."""
    return bool(re.match(r'^\d{2}\.[A-Za-zğüşıöçĞÜŞİÖÇ]+\.\d{2}\s+[PÇCPSPÇC]\w+', line))


def format_date_range(date1, date2):
    if date1 and date2:
        return f"{date1} - {date2}"
    if date1:
        return date1
    return ""
