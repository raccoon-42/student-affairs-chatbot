from datetime import datetime

from preprocessing.extraction import (
    parse_date,
    extract_event_type,
    extract_academic_period,
    is_date_line,
    format_date_range,
)


def test_parse_turkish_date():
    assert parse_date("14.Şub.25") == datetime(2025, 2, 14)


def test_parse_date_strips_day_name():
    assert parse_date("14.Şub.25 Cuma") == datetime(2025, 2, 14)


def test_parse_date_invalid_returns_none():
    assert parse_date("not a date") is None


def test_event_type_deadline_beats_course():
    # "ders" is in the query too, but "son tarih" is more specific and wins
    assert extract_event_type("dersten çekilme için son tarih nedir") == "deadline"


def test_event_type_default_for_chunks():
    assert extract_event_type("bilinmeyen bir satır", default="event") == "event"


def test_event_type_no_match_means_no_filter():
    assert extract_event_type("bilinmeyen bir satır") is None


def test_academic_period():
    assert extract_academic_period("GÜZ YARIYILI") == "fall"
    assert extract_academic_period("bahar dönemi sınavları") == "spring"
    assert extract_academic_period("yaz okulu kayıtları") == "summer"


def test_yazili_is_not_summer():
    # 'yazılı sınav' (written exam) must not match 'yaz' (summer)
    assert extract_academic_period("yazılı sınav sonuçları") is None


def test_is_date_line():
    assert is_date_line("14.Şub.25 Cuma Derslerin başlaması")
    assert not is_date_line("TITLE: GÜZ YARIYILI")


def test_format_date_range():
    assert format_date_range("1.Oca.25", "5.Oca.25") == "1.Oca.25 - 5.Oca.25"
    assert format_date_range("1.Oca.25", None) == "1.Oca.25"
    assert format_date_range(None, None) == ""
