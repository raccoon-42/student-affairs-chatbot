"""Scrape a department's people (academic staff, assistants, admin).

Pilot: ceng. Department sites share the same WordPress theme, so the
selectors here should transfer; DEPARTMENTS grows one line per site as
the rollout confirms that. The roster page (/people/) yields each
person's name, academic title and section; the profile page adds the
contact block (theme-structured stm-contact-details items: office,
email, phone, website) and the first bio paragraph. Publication lists
are deliberately not taken — they'd drown retrieval and go stale.

No LLM step: extraction is deterministic, so the scraper writes the
processed JSON directly (like the FAQ corpus). Output:
preprocessing/data/processed/people/<dept>.json

Run: uv run python -m preprocessing.scrapers.people_scraper [dept ...]
(no args = every configured department)
"""
import json
import re
import sys
import time

import requests
from bs4 import BeautifulSoup

from config import settings

DEPARTMENTS = {
    "ceng": {
        "name": "Bilgisayar Mühendisliği",
        "people_url": "https://ceng.iyte.edu.tr/people/",
    },
}

# roster tab-panel heading -> role category
SECTIONS = {
    "Academic Members": "akademik",
    "Research Assistants": "arastirma-gorevlisi",
    "Administrative Staff": "idari",
}

# sub-headings INSIDE a panel that end scraping for that panel — retired
# faculty are listed under the Academic Members tab but must not appear
# in "who teaches here" answers
SUBSECTIONS_SKIPPED = {"Retired Faculty Members"}

OUTPUT_DIR = settings.ROOT / "preprocessing" / "data" / "processed" / "people"

# stm-contact-details__item_type_<key> -> output field; the department
# postal address (type_address) is the same for everyone, so it's skipped
CONTACT_TYPES = {"email": "e-posta", "tel": "telefon", "roomno": "ofis", "url": "web"}


_session = requests.Session()


def fetch(url):
    # the site starts dropping connections under ~46 rapid sequential
    # requests — pace politely and retry timeouts with backoff
    for attempt in range(3):
        try:
            resp = _session.get(url, timeout=30)
            resp.raise_for_status()
            time.sleep(0.5)
            return BeautifulSoup(resp.text, "html.parser")
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == 2:
                raise
            wait = 5 * (attempt + 1)
            print(f"  {e.__class__.__name__} on {url}, retrying in {wait}s...")
            time.sleep(wait)


def extract_roster(soup):
    """Yield (name, title, profile_url, section) per roster card.

    The roster is a WPBakery tab set: one vc_tta-panel per section, the
    section name in the panel's own heading, one stm-teacher card per
    person (name and academic title are card fields — profile pages
    with custom layouts don't carry them reliably)."""
    for panel in soup.select(".vc_tta-panel"):
        heading = panel.select_one(".vc_tta-panel-title")
        section = SECTIONS.get(" ".join(heading.get_text().split())) if heading else None
        if section is None:
            continue
        skipping = False
        for element in panel.find_all(["h4", "div"]):
            if element.name == "h4":
                if " ".join(element.get_text().split()) in SUBSECTIONS_SKIPPED:
                    skipping = True  # everything below this sub-heading is out
                continue
            if skipping or "stm-teacher" not in (element.get("class") or []):
                continue
            name_el = element.select_one(".stm-teacher__name")
            if name_el is None:
                continue
            name = " ".join(name_el.get_text().split())
            role_el = element.select_one(".stm-teacher-title-ad-role")
            title = " ".join(role_el.get_text().split()) if role_el else None
            link = name_el.find("a")
            yield name, title or None, link["href"] if link else None, section


def extract_profile(soup):
    """(contact dict, bio) from a profile page — name and title come
    from the roster card, which every layout variant fills in."""
    contact = {}
    for item in soup.select(".stm-contact-details__items li"):
        classes = " ".join(item.get("class", []))
        kind = next((label for key, label in CONTACT_TYPES.items() if f"_type_{key}" in classes), None)
        value = " ".join(item.get_text().split())
        if kind and value:  # a person can list several urls — keep them all
            contact[kind] = f"{contact[kind]}, {value}" if kind in contact else value

    # first substantial text paragraph = the bio; it precedes the
    # publication lists in every layout variant seen so far
    bio = None
    for p in soup.select(".wpb_text_column p"):
        text = " ".join(p.get_text().split())
        if len(text) > 120:
            bio = text
            break
    return contact, bio


def scrape_department(dept, config):
    print(f"Fetching roster: {config['people_url']}")
    roster = list(extract_roster(fetch(config["people_url"])))
    if not roster:
        raise ValueError(f"{dept}: empty roster — selectors or section headings changed?")

    people = []
    for name, title, url, section in roster:
        contact, bio = extract_profile(fetch(url)) if url else ({}, None)
        print(f"  [{section}] {name}" + (f" — {title}" if title else ""))
        people.append({
            "name": name,
            "title": title,
            "role": section,
            "department": config["name"],
            "contact": contact,
            "bio": bio,
            "source_url": url or config["people_url"],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{dept}.json"
    output.write_text(json.dumps(people, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(people)} people -> {output}")


def main():
    departments = sys.argv[1:] or list(DEPARTMENTS)
    unknown = [d for d in departments if d not in DEPARTMENTS]
    if unknown:
        sys.exit(f"Unknown department(s) {unknown}; configured: {', '.join(DEPARTMENTS)}")
    for dept in departments:
        scrape_department(dept, DEPARTMENTS[dept])


if __name__ == "__main__":
    main()
