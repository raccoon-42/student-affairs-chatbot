"""Scrape course catalogs from the department sites.

Each graduate catalog page (MS / PhD courses) holds tables with the
header "Course Code | Course Name | Description | Prerequisite(s)"; other
tables on the same pages (weekly topics, program outcomes) don't match the
header and are skipped. The undergraduate courses page only links a program
PDF (since 2026-10), parsed per course header line. A course appearing in
several catalogs (MS + PhD share the 5xx pool) is merged into one entry
carrying every level.

Writes preprocessing/data/processed/courses/<dept>.json directly — the
extraction is deterministic, so like the people scraper there is no LLM
step and the scraped JSON is the processed corpus.

Usage: python -m preprocessing.scrapers.courses_scraper [dept ...]
"""
import io
import json
import re
import sys
from collections import Counter
from urllib.parse import urljoin

import pdfplumber
from bs4 import BeautifulSoup

from config import settings
from preprocessing.scrapers.fetch import fetch

# one entry per department for rollout, like the people scraper
DEPARTMENTS = {
    "ceng": {
        "department": "Bilgisayar Mühendisliği",
        "prefixes": ("CENG",),  # the program PDF also lists common MATH/PHYS/TURK courses
        "catalogs": {
            "yukseklisans": ["https://ceng.iyte.edu.tr/education/ms-programs/ceng/courses/",
                             "https://ceng.iyte.edu.tr/education/ms-programs/seds/courses/"],
            "doktora": ["https://ceng.iyte.edu.tr/education/phd-program/courses/"],
        },
        # page -> linked PDF; the PDF moves to a new upload path on every revision
        "program_documents": {
            "lisans": "https://ceng.iyte.edu.tr/education/undergraduate-program/courses/",
        },
        # the only page listing credits/AKTS; electives have no row in its
        # semester tables, so they stay without
        "curriculum": "https://ceng.iyte.edu.tr/tr/egitim/lisans-programi/lisans-egitim-plani-2018-ve-sonrasi/",
    },
}

OUTPUT_DIR = settings.ROOT / "preprocessing" / "data" / "processed" / "courses"

HEADER = ["course code", "course name", "description", "prerequisite(s)"]
CURRICULUM_HEADER = ["bölüm", "ders kodu", "ders adı", "saatler", "ön koşullar", "krediler", "akts"]
SEMESTER = re.compile(r"^\s*(\d+)\. Dönem")
# program PDF header line per course: "CENG 322 Operating Systems (3+2)4 8 Ects",
# also seen: "OHS101Occupational ... (1+0)1 Ects 1" and "(4-0)4 – 5 Ects"
COURSE_LINE = re.compile(
    r"^([A-Z]{2,5}) ?(\d{3}[A-Z]?)(?![a-z])\s*(.+?)\s*\((\d+)[+-](\d+)\)\s*(\d+)\s*(?:[–-]\s*)?"
    r"(?:(\d+)\s*Ects|Ects\s*(\d+))", re.M)

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


def extract_curriculum(html):
    """Yield (code, hours, credits, ects, semester) per course row of the
    curriculum's semester tables. Total rows and elective placeholders
    ("Sınırlı Seçmeli I") have no course code and are skipped."""
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if not rows or [_cell_text(c).lower() for c in rows[0].find_all(["th", "td"])] != CURRICULUM_HEADER:
            continue
        heading = table.find_previous(string=SEMESTER)
        semester = int(SEMESTER.match(heading).group(1)) if heading else None
        for row in rows[1:]:
            cells = [_cell_text(c) for c in row.find_all(["th", "td"])]
            if len(cells) < 7 or not cells[0] or not cells[1]:
                continue
            yield f"{cells[0]} {cells[1]}", cells[3].strip("()"), cells[5], cells[6], semester


def program_document_url(html, page_url):
    link = BeautifulSoup(html, "html.parser").find("a", href=re.compile(r"\.pdf$", re.I))
    if link is None:
        raise ValueError(f"{page_url}: no program PDF linked — page layout changed?")
    return urljoin(page_url, link["href"])


def without_running_lines(pages):
    """Join page texts minus lines found on most pages (the address footer),
    which would otherwise land inside descriptions that cross a page break."""
    counts = Counter(line for lines in pages for line in set(lines))
    running = {line for line, n in counts.items() if n > len(pages) / 2}
    return "\n".join(line for lines in pages for line in lines if line not in running)


def pdf_text(content):
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        return without_running_lines([(page.extract_text() or "").splitlines() for page in pdf.pages])


def parse_program_document(text, prefixes):
    """Yield (code, name, hours, credits, ects, description, prerequisites)
    per course whose subject is the department's own; the description is
    the block before "Learning Outcomes:"."""
    heads = list(COURSE_LINE.finditer(text))
    for head, after in zip(heads, heads[1:] + [None]):
        subject, number, name, theory, practice, credits, ects, ects_after = head.groups()
        if subject not in prefixes:
            continue
        block = text[head.end(): after.start() if after else len(text)]
        description = re.split(r"Learning Outcomes:|Prerequisite:", block)[0]
        description = " ".join(line for line in description.splitlines() if not line.isupper())
        prerequisite = re.search(r"^Prerequisite:\s*(.+)$", block, re.M)
        yield (f"{subject} {number}", name.strip(), f"{theory}+{practice}", credits,
               ects or ects_after, " ".join(description.split())[:900],
               prerequisite.group(1).strip() if prerequisite else None)


def carry_over(previous, by_code, level, found):
    """Re-add courses of `level` from the previous scrape that the new source
    lacks: a course dropped from the program PDF isn't necessarily retired
    (CENG 323/418 are still on the curriculum's elective list). Delete one
    from the JSON by hand once it really is gone."""
    kept = []
    for old in previous:
        if level in old["levels"] and old["code"] not in found:
            entry = by_code.setdefault(old["code"], {**old, "levels": []})
            if level not in entry["levels"]:
                entry["levels"].append(level)
            kept.append(old["code"])
    return kept


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

    path = OUTPUT_DIR / f"{dept}.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    for level, page_url in config.get("program_documents", {}).items():
        url = program_document_url(fetch(page_url).text, page_url)
        print(f"[{level}] {url}")
        found = set()
        for code, name, hours, credits, ects, description, prerequisites in \
                parse_program_document(pdf_text(fetch(url).content), config["prefixes"]):
            found.add(code)
            entry = by_code.setdefault(code, {
                "code": code,
                "name": name,
                "description": description,
                "prerequisites": prerequisites,
                "levels": [],
                "department": config["department"],
                "source_url": url,
            })
            entry.update(hours=hours, credits=credits, ects=ects)
            if level not in entry["levels"]:
                entry["levels"].append(level)
        print(f"  {len(found)} courses")
        if len(found) < 20:  # 68 CENG courses in the 2026-10 PDF
            raise ValueError(f"{url}: only {len(found)} courses parsed — PDF layout changed?")
        kept = carry_over(previous, by_code, level, found)
        if kept:
            print(f"  kept from the previous scrape (not in the PDF): {', '.join(kept)}")

    if config.get("curriculum"):
        url = config["curriculum"]
        print(f"[curriculum] {url}")
        matched = 0
        for code, hours, credits, ects, semester in extract_curriculum(fetch(url).text):
            entry = by_code.get(code)
            if entry:  # common courses (MATH, PHYS, TURK...) aren't in the dept catalog
                entry["semester"] = semester
                # the 2026-10 program PDF is newer than this 2018 plan
                for key, value in (("hours", hours), ("credits", credits), ("ects", ects)):
                    entry.setdefault(key, value)
                matched += 1
        print(f"  semester for {matched} courses")
        if not matched:
            raise ValueError(f"{url}: no curriculum rows matched — page layout changed?")

    courses = sorted(by_code.values(), key=lambda c: c["code"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
