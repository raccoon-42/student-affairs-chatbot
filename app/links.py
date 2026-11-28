"""Removes links the model made up.

Prompt rule 7 lets a URL into an answer only when a reference chunk has it
verbatim. The model breaks that now and then, mostly with a plausible
university address that no chunk contains ("https://oidb.iyte.edu.tr",
"https://www.iyte.edu.tr"): 7 of 27 logged judge failures by 2026-11-26.
Also caught: the same address without "https://" ("(oidb.iyte.edu.tr)"),
and a link whose target is no address at all ("(mezuniyet-durumu-formu)",
"(Öğrenci Ders Kayıt Kılavuzu)").
A link whose address appears in none of the turn's chunks is dropped and a
markdown link keeps its text, so "[Öğrenci İşleri](https://oidb.iyte.edu.tr)"
reads "Öğrenci İşleri". E-mail addresses are left alone. No model call.
"""
import re
from typing import List, Tuple

# link text that is only a number is a citation marker followed by a
# parenthesis ("[38](PDF)"), not a link. A "[" that starts no link matches
# too, up to where the next link can start, and is kept: tried again from
# each "[" of a long run, the rest of the answer was rescanned once per "["
_MARKDOWN = re.compile(r"\[(?!\d+\])([^\]]*)(?:\]\(([^)\n]+)\)|\]\([^)\n]*\])?")
_URL = re.compile(r"https?://[^\s)\]>\"']+")
# a web address written without its scheme; not after "@" (e-mail), "/" or
# ":" (inside a URL), "." or a letter (inside a longer name)
_DOMAIN = re.compile(r"(?<![\w@/:.-])(?:[a-z0-9-]+\.)+(?:edu|gov|com|org|net)(?:\.tr)?\b(?:/[^\s)\]>\"']*)?"
                     r"|(?<![\w@/:.-])(?:[a-z0-9-]+\.)+tr\b(?:/[^\s)\]>\"']*)?", re.IGNORECASE)
_TRAILING = ".,;:!?"


def _key(address):
    """"http://turnaportal.ua.gov.tr/" in a chunk backs "turnaportal.ua.gov.tr" in an answer."""
    address = address.rstrip(_TRAILING).rstrip("/").lower()
    return re.sub(r"^https?://", "", address)


def _is_address(text):
    text = text.strip()
    return bool(_URL.fullmatch(text) or _DOMAIN.fullmatch(text))


def check_links(answer: str, chunks: List[str]) -> Tuple[str, List[str]]:
    """The answer without links no chunk contains, and the links removed."""
    # whole addresses only: "uio.iyte.edu.tr" inside a chunk's longer
    # "https://uio.iyte.edu.tr/wp-content/..." link doesn't back the bare root
    text_of_chunks = "\n".join(chunks)
    known = {_key(address) for address in _URL.findall(text_of_chunks) + _DOMAIN.findall(text_of_chunks)}

    def backed(address):
        return _key(address) in known

    removed = []

    def markdown(match):
        if match.group(2) is None:
            return match.group(0)  # no link
        text, target = match.group(1), match.group(2).strip()
        if (_URL.fullmatch(target) or _DOMAIN.fullmatch(target)) and backed(target):
            return match.group(0)
        removed.append(target)
        # a link whose text is the address itself goes entirely
        return "" if _is_address(text) else text

    def bare(match):
        address = match.group(0)
        if backed(address):
            return address
        kept = address.rstrip(_TRAILING)
        removed.append(kept)
        return address[len(kept):]  # the sentence keeps its full stop

    answer = _MARKDOWN.sub(markdown, answer)
    answer = _DOMAIN.sub(bare, _URL.sub(bare, answer))
    if removed:
        answer = re.sub(r"\(\s*\)", "", answer)  # "Öğrenci İşleri'ne ()" left by a removed link
        answer = re.sub(r"(?<=\S)[ \t]{2,}", " ", answer)  # not leading spaces: nested lists
        answer = re.sub(r"(?<=\S) +([.,;:!?])", r"\1", answer)
    return answer, removed
