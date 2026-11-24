from preprocessing.scrapers.courses_scraper import (carry_over, extract_curriculum,
                                                    parse_program_document, program_document_url,
                                                    without_running_lines)

# trimmed from the CENG "Lisans Eğitim Planı 2018 ve Sonrası" page
CURRICULUM = """
<h3>5. Dönem</h3>
<table>
  <tr><th>Bölüm</th><th>Ders Kodu</th><th>Ders Adı</th><th>Saatler</th>
      <th>Ön Koşullar</th><th>Krediler</th><th>AKTS</th></tr>
  <tr><td></td><td></td><td></td><td></td><td></td>
      <td>Dönem Kredisi: 16</td><td>Toplam AKTS: 29</td></tr>
  <tr><td>CENG</td><td>311</td><td>Bilgisayar Mimarisi</td><td>(3+2)</td>
      <td>CENG214</td><td>4</td><td>8</td></tr>
  <tr><td></td><td></td><td>Sınırlı Seçmeli I</td><td>(3+0)</td><td></td><td>3</td><td>6</td></tr>
</table>
<h3>Sınırlı Seçmeli Dersler</h3>
<table>
  <tr><th>Course Code</th><th>Ders Adı</th><th>Ön Koşul(lar)</th></tr>
  <tr><td>CENG 323</td><td>Proje Yönetimi</td><td></td></tr>
</table>
"""

# shaped like the text of the 2026-10 undergraduate program PDF, including
# its irregular header lines (glued code, "Ects 1", "(4-0)4 – 5 Ects")
PROGRAM = """IZMIR INSTITUTE OF TECHNOLOGY
UNDERGRADUATE PROGRAM
MATH 141 Basic Calculus I (3+2)4 5 Ects
Limits and continuity.
Learning Outcomes:
1.To be able to compute limits.
CENG 311 Computer Architecture (3+2)4 8 Ects
Basic computer organization and design.
CPU organization.
Learning Outcomes:
1.To be able to design microprocessors.
Prerequisite: CENG 214
TECHNICAL ELECTIVE COURSES
CENG 435 Database System Design and Implementation (3+0)3 6 Ects
Physical database design.
CENG101Odd Glued Header (1+0)1 Ects 1
Glued.
CENG 102 Dash Format (4-0)4 – 5 Ects
Dashed."""


def test_curriculum_rows_carry_hours_credits_ects_and_semester():
    assert list(extract_curriculum(CURRICULUM)) == [("CENG 311", "3+2", "4", "8", 5)]


def test_program_document_yields_own_courses_with_credits():
    courses = list(parse_program_document(PROGRAM, ("CENG",)))
    assert courses == [
        ("CENG 311", "Computer Architecture", "3+2", "4", "8",
         "Basic computer organization and design. CPU organization.", "CENG 214"),
        ("CENG 435", "Database System Design and Implementation", "3+0", "3", "6",
         "Physical database design.", None),
        ("CENG 101", "Odd Glued Header", "1+0", "1", "1", "Glued.", None),
        ("CENG 102", "Dash Format", "4+0", "4", "5", "Dashed.", None),
    ]  # MATH 141 is a common course: not CENG's to list


def test_running_footer_lines_are_dropped():
    footer = ["Izmir Institute of Technology", "Tel: +90 232 750 7860"]
    pages = [["CENG 311 ...", "first half"] + footer, ["second half"] + footer, ["x"] + footer]
    assert without_running_lines(pages) == "CENG 311 ...\nfirst half\nsecond half\nx"


def test_program_document_url_is_looked_up_on_the_page():
    html = '<main><p>See <a href="/wp-content/uploads/2026/10/PROGRAM.pdf">document</a></p></main>'
    assert (program_document_url(html, "https://ceng.iyte.edu.tr/education/undergraduate-program/courses/")
            == "https://ceng.iyte.edu.tr/wp-content/uploads/2026/10/PROGRAM.pdf")


def test_courses_missing_from_the_new_source_carry_over():
    previous = [{"code": "CENG 323", "name": "Project Management", "levels": ["lisans"]},
                {"code": "CENG 311", "name": "Old Name", "levels": ["lisans"]},
                {"code": "CENG 524", "name": "Grad Only", "levels": ["yukseklisans"]}]
    by_code = {"CENG 311": {"code": "CENG 311", "name": "Computer Architecture", "levels": ["lisans"]}}
    assert carry_over(previous, by_code, "lisans", found={"CENG 311"}) == ["CENG 323"]
    assert by_code["CENG 323"]["levels"] == ["lisans"]
    assert by_code["CENG 311"]["name"] == "Computer Architecture"  # fresh data wins
    assert "CENG 524" not in by_code  # other levels come from their own catalogs
