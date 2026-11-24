"""The guides scraper's link discovery, offline on HTML shaped like UİO's."""
import pytest

from preprocessing.scrapers.guides_scraper import (ANNOUNCEMENT_MENU_TEXT, PAGES, menu_link,
                                                   newest_erasmus_call)

MENU = """<nav><a href="https://uio.iyte.edu.tr/erasmus-giden-ogrenci/">Erasmus+ Öğrenim Hareketliliği Giden Öğrenci</a>
<a href="https://uio.iyte.edu.tr/2025-erasmus-k-131-ogrenim-duyurusu/">Erasmus + KA131
  Öğrenim Hareketliliği İlanı</a></nav>"""

ANNOUNCEMENTS = """<article>
<a href="https://uio.iyte.edu.tr/duyuru/2025-erasmus-ka-131-ogrenim-hareketliligi-basvurulari/">2025 ilanı</a>
<a href="https://uio.iyte.edu.tr/duyuru/2026-erasmus-ka-131-ogrenim-hareketliligi-basvurulari/">2026 ilanı</a>
<a href="https://outlook.office.com/book/RANDEVU/">RANDEVU LİNK</a></article>"""


def test_the_announcement_page_is_found_by_its_menu_text():
    assert menu_link(MENU, ANNOUNCEMENT_MENU_TEXT).endswith("/2025-erasmus-k-131-ogrenim-duyurusu/")


def test_the_newest_years_call_is_the_one_indexed():
    # the call moves to a new URL every year; last year's must not win
    assert newest_erasmus_call(ANNOUNCEMENTS).endswith("/duyuru/2026-erasmus-ka-131-ogrenim-hareketliligi-basvurulari/")


def test_a_missing_call_fails_loudly():
    with pytest.raises(ValueError):
        newest_erasmus_call("<article><a href='/duyuru/kampus-kapanis/'>x</a></article>")


def test_page_slugs_stay_unique():
    slugs = [url.rstrip("/").rsplit("/", 1)[-1] for _, url in PAGES]
    assert len(set(slugs)) == len(slugs)
