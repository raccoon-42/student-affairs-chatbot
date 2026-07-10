"""Scrape course catalogs from the department sites.

Each catalog page (undergraduate / MS / PhD courses) holds tables with the
header "Course Code | Course Name | Description | Prerequisite(s)"; other
tables on the same pages (weekly topics, program outcomes) don't match the
header and are skipped. A course appearing in several catalogs (MS + PhD
share the 5xx pool) is merged into one entry carrying every level.

Writes preprocessing/data/processed/courses/<dept>.json directly — the
extraction is deterministic, so like the people scraper there is no LLM
step and the scraped JSON is the processed corpus.

Usage: python -m preprocessing.scrapers.courses_scraper [dept ...]
"""
import json
import re
import sys

from bs4 import BeautifulSoup

from config import settings
from preprocessing.scrapers.fetch import fetch

# one entry per department for rollout, like the people scraper
DEPARTMENTS = {
    "ceng": {
        "department": "Bilgisayar Mühendisliği",
        "catalogs": {
            "lisans": ["https://ceng.iyte.edu.tr/education/undergraduate-program/courses/"],
            "yukseklisans": ["https://ceng.iyte.edu.tr/education/ms-programs/ceng/courses/",
                             "https://ceng.iyte.edu.tr/education/ms-programs/seds/courses/"],
            "doktora": ["https://ceng.iyte.edu.tr/education/phd-program/courses/"],
        },
    },
}

OUTPUT_DIR = settings.ROOT / "preprocessing" / "data" / "processed" / "courses"

HEADER = ["course code", "course name", "description", "prerequisite(s)"]

CODE_PATTERN = re.compile(r"^[A-Z]{2,5} ?\d{3}[A-Z]?$")

# the description cell dumps the whole detail page in some rows —
# objectives, reading lists, weekly topics, grading matrices. The catalog
# description proper is everything before the first of these markers.
DESCRIPTION_CUTOFFS = ("Course Objectives", "Recommended or Required Reading",
                       "Learning Outcomes", "LECTURES & LABS", "Week Topics",
                       "Topics Introduction", "Grading:", "References:")


def _cell_text(cell):
    return " ".join(cell.get_text().split())


def _description(cell):
    text = _cell_text(cell)
    positions = [p for marker in DESCRIPTION_CUTOFFS if (p := text.find(marker)) > 0]
    if positions:
        text = text[:min(positions)]
    return text.rstrip()[:900]


def extract_courses(html, page_url):
    """Yield (code, name, description, prerequisites, detail_url) per row
    of every catalog table on the page."""
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows or [_cell_text(c).lower() for c in rows[0].find_all(["th", "td"])] != HEADER:
            continue
        for row in rows[1:]:
            # direct children only: expanded rows nest the detail page's
            # own tables, whose cells would shift the column positions
            cells = row.find_all("td", recursive=False)
            if len(cells) < 4:
                continue
            code = _cell_text(cells[0])
            if not CODE_PATTERN.match(code):
                continue
            link = row.find("a", href=re.compile("/courses/"))
            prerequisites = _cell_text(cells[3]).strip("-–— ")
            if len(prerequisites) > 100:  # nested junk, not a prerequisite line
                prerequisites = ""
            yield (code, _cell_text(cells[1]), _description(cells[2]),
                   prerequisites or None, link["href"] if link else page_url)


def scrape_department(dept, config):
    by_code = {}
    for level, urls in config["catalogs"].items():
        for url in urls:
            print(f"[{level}] {url}")
            found = 0
            for code, name, description, prerequisites, detail_url in \
                    extract_courses(fetch(url).text, url):
                found += 1
                entry = by_code.setdefault(code, {
                    "code": code,
                    "name": name,
                    "description": description,
                    "prerequisites": prerequisites,
                    "levels": [],
                    "department": config["department"],
                    "source_url": detail_url,
                })
                if level not in entry["levels"]:
                    entry["levels"].append(level)
            print(f"  {found} courses")
            if not found:
                raise ValueError(f"{url}: no catalog tables matched — page layout changed?")

    courses = sorted(by_code.values(), key=lambda c: c["code"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{dept}.json"
    path.write_text(json.dumps(courses, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{dept}: {len(courses)} courses -> {path}")


def main():
    departments = sys.argv[1:] or list(DEPARTMENTS)
    unknown = [d for d in departments if d not in DEPARTMENTS]
    if unknown:
        sys.exit(f"Unknown department(s) {unknown}; configured: {', '.join(DEPARTMENTS)}")
    for dept in departments:
        scrape_department(dept, DEPARTMENTS[dept])


if __name__ == "__main__":
    main()
