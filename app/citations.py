"""Checks the [n] markers in an answer against the chunks they point to.

The model cites reference chunks as [n]. Now and then it numbers them like
paper footnotes instead: the facts of chunks [36]-[38] come out as [1]-[3]
with a "[1] [2] [3]" line at the end, and the UI shows chips for whatever
chunks [1]-[3] were. Checks that catch it without another model call:
- a comma list of markers ("[26, 27]") becomes separate ones ("[26][27]"),
  the only form the UI chips and the judge read;
- a last line of nothing but markers is removed;
- a marker with no chunk behind it is removed;
- in an answer shaped like that (markers run [1]..[k], or a footer line),
  a marker whose chunk shares no distinctive word with the text it
  follows is removed. Only there: elsewhere the word test misjudges
  translations (the course catalog is English, answers are Turkish).
No chip beats a wrong one.
"""
import re
from collections import Counter
from typing import List, Tuple

from preprocessing.indexing.bm25 import tokenize

_RUN = re.compile(r"( ?)((?:\[\d+\])+)")  # one or more markers in a row: "[1][4]"
_MARKER = re.compile(r"\[(\d+)\]")
_FOOTER = re.compile(r"[ \t]*(?:\[\d+\][ \t]*)+")  # a line of nothing but markers
_LIST = re.compile(r"\[\d+(?:\s*,\s*\d+)+\]")  # "[26, 27]"
# shorter text before a marker ("Evet [3].") is too little to judge by shared words
MIN_WORDS = 4
# a word in more than half of the turn's chunks ("öğrenci", "iyte") says
# nothing about which chunk was cited. Needs enough chunks to tell; a
# production turn carries ~40
COMMON_SHARE = 0.5
MIN_CHUNKS_FOR_COMMON = 10


def check_citations(answer: str, chunks: List[str]) -> Tuple[str, List[int]]:
    """The answer without unsupported markers, and the marker numbers removed.
    chunks[n - 1] is the text marker [n] points to."""
    removed = []
    answer = _LIST.sub(lambda m: "".join(f"[{n}]" for n in re.findall(r"\d+", m.group(0))), answer)
    cited = sorted({int(n) for n in _MARKER.findall(answer)})
    # renumbering cites [1]..[k] in order of use; a normal answer cites the
    # numbers of the chunks it used
    renumbered = bool(cited) and cited == list(range(1, len(cited) + 1))
    body, _, last = answer.rstrip().rpartition("\n")
    if body.strip() and _FOOTER.fullmatch(last):
        removed += [int(n) for n in _MARKER.findall(last)]
        answer = body.rstrip()
        renumbered = True

    words_of = [set(tokenize(chunk)) for chunk in chunks]
    common = set()
    if len(chunks) >= MIN_CHUNKS_FOR_COMMON:
        counts = Counter(word for words in words_of for word in words)
        common = {word for word, count in counts.items() if count > COMMON_SHARE * len(chunks)}

    def supported(n, words):
        if not 1 <= n <= len(chunks):
            return False  # no such chunk: never a valid citation
        if not renumbered or len(words) < MIN_WORDS:
            return True
        return bool((words & words_of[n - 1]) - common)

    lines = []
    for line in answer.split("\n"):
        kept_line, start = [], 0
        for run in _RUN.finditer(line):
            # a marker covers the text since the previous marker on its line
            words = set(tokenize(line[start:run.start()]))
            numbers = [int(n) for n in _MARKER.findall(run.group(2))]
            kept = [n for n in numbers if supported(n, words)]
            removed += [n for n in numbers if n not in kept]
            kept_line.append(line[start:run.start()])
            if kept:
                kept_line.append(run.group(1) + "".join(f"[{n}]" for n in kept))
            start = run.end()
        kept_line.append(line[start:])
        lines.append("".join(kept_line))
    return "\n".join(lines), removed


# a list line ("- Fizik", "1. Kayıt") or a bold heading line ("**Fen Fakültesi**")
_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S")
_HEADING = re.compile(r"^\s*\*\*[^*]+\*\*:?\s*$")
_LEAD = re.compile(r"(\[\d+\])*\s*:\s*$")  # a line introducing a list ends with ":"
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=\S)")
# shared words a source needs before its marker is added: one shared
# word ("bölüm") says little, two distinctive ones ("voleybol", "16.00") do
MIN_SHARED = 2
# sentences that need no marker, so get none: a question back, sending the
# student somewhere ("SKS'ye danışmanızı öneririm"), saying the data lacks
# something. A marker there makes the judge check them against a source
# that doesn't say them, failing an answer that passed
_NOT_A_FACT = re.compile(
    r"\?\s*$|danış|iletişime geç|öneri|teyit|takip e|kontrol et|başvurabilir|bilgi alabilir|"
    r"öğrenebilir|sorabilir|en doğrusu|\bbugün|"  # today's date is the system's, not a source's
    r"\w+(?:abil|ebil)irim\b|"  # an offer: "gruplayabilirim", "belirtebilirim"
    r"\b(?:referansta|verilerde|elimde(?:ki)?)\b[^.;]*\b(?:yok|yer almıyor|bulunmuyor|değil)\b|"
    r"aktaramıyorum|doğrulayamadım|bilmiyorum",
    re.IGNORECASE)


def _concrete(text):
    """A fact worth a marker: a number, or a name (a capitalized word after
    the first). "Kesin bilgi için danışabilirsin." is none. Judged clause by
    clause: "Başvurular 2 Nisan 2026'da kapandı; Erasmus birimine
    danışabilirsin." still carries a date that needs one."""
    for clause in re.split(r";", text):
        if _NOT_A_FACT.search(clause):
            continue
        words = re.sub(r"[*_`]", "", clause).split()
        if any(ch.isdigit() for ch in clause) or any(w[:1].isupper() for w in words[1:]):
            return True
    return False


def add_missing_markers(answer: str, chunks: List[str]) -> Tuple[str, List[int]]:
    """Facts no marker covers get the marker of a source the answer already
    cites, when one clearly holds them; the markers added. The model now
    and then cites a source once and leaves the rest bare: a list under an
    unmarked ":" line, a sentence after the paragraph's marker. Coverage
    follows the judge: a marker covers its sentence and the unmarked ones
    before it in its paragraph; one on a ":" line covers the list after it,
    as does one at the end of the list. Only cited sources are candidates:
    no chip beats a wrong one."""
    cited = sorted({int(n) for n in _MARKER.findall(answer) if 1 <= int(n) <= len(chunks)})
    if not cited:
        return answer, []
    words_of = {n: set(tokenize(chunks[n - 1])) for n in cited}
    common = set()
    if len(chunks) >= MIN_CHUNKS_FOR_COMMON:
        counts = Counter(word for chunk in chunks for word in set(tokenize(chunk)))
        common = {word for word, count in counts.items() if count > COMMON_SHARE * len(chunks)}

    def source_for(text):
        if not _concrete(text):
            return None
        words = set(tokenize(_MARKER.sub("", text))) - common
        # what tells sources apart is the words only one of them has: two
        # gym timetables share every day and hour, only one says "voleybol"
        own = {n: words & words_of[n] - set().union(*(words_of[m] for m in cited if m != n))
               for n in cited}
        scores = sorted(((len(own[n]), n) for n in cited), reverse=True)
        best, n = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else 0
        if best > runner_up and len(words & words_of[n]) >= MIN_SHARED:
            return n
        return None

    lines = answer.split("\n")
    added = []

    def is_list(line):
        return bool(_ITEM.match(line) or _HEADING.match(line))

    def list_end(i):
        """Index after the list starting at i, blank lines between its parts included."""
        j = i
        while j < len(lines) and (is_list(lines[j]) or not lines[j].strip()):
            j += 1
        while j > i and not lines[j - 1].strip():
            j -= 1
        return j

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        lead = None
        if not is_list(line) and _LEAD.search(line):
            k = i + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines) and is_list(lines[k]):
                lead, i = i, k
        if is_list(lines[i]):
            end = list_end(i)
            items = [n for n in range(i, end) if lines[n].strip()]
            marked = [n for n in items if _MARKER.search(lines[n])]
            covered = (lead is not None and _MARKER.search(lines[lead])) or (marked and marked[-1] == items[-1])
            if not covered and not marked:
                text = "\n".join(lines[n] for n in ([lead] if lead is not None else []) + items)
                n = source_for(text)
                if n:
                    added.append(n)
                    if lead is not None:
                        lines[lead] = re.sub(r"\s*:\s*$", f" [{n}]:", lines[lead])
                    else:
                        lines[items[-1]] = lines[items[-1]].rstrip() + f" [{n}]"
            i = end
            continue
        # a prose paragraph: its lines up to a blank line or a list
        end = i
        while end < len(lines) and lines[end].strip() and not is_list(lines[end]):
            end += 1
        paragraph = " ".join(lines[i:end])
        sentences = _SENTENCE.split(paragraph)
        last_marked = max((k for k, sentence in enumerate(sentences) if _MARKER.search(sentence)),
                          default=-1)
        facts = [sentence for sentence in sentences[last_marked + 1:] if _concrete(sentence)]
        # an unmarked paragraph gets one marker, on its last fact: it covers
        # the facts before it. After the paragraph's marker, each fact its own
        units = [(" ".join(facts), facts[-1])] if last_marked < 0 and facts else \
            [(sentence, sentence) for sentence in facts]
        for text, target in units:
            n = source_for(text)
            home = next((k for k in range(i, end) if target in lines[k]), None)
            if n and home is not None:  # a sentence split over lines is left alone
                added.append(n)
                marked = re.sub(r"\s*([.!?:]?)$", f" [{n}]\\1", target.rstrip(), count=1)
                lines[home] = lines[home].replace(target.rstrip(), marked, 1)
        i = end
    return "\n".join(lines), added


_NUMBER = re.compile(r"\d+(?:[.,:/]\d+)*")


def _holds(chunk, number):
    """The number as a whole in the chunk: "107" not inside "1070" or "5.107"."""
    return re.search(rf"(?<!\d)(?<!\d[.,:/]){re.escape(number)}(?!\d)(?![.,:/]\d)", chunk) is not None


def correct_markers(answer: str, chunks: List[str]) -> Tuple[str, List[Tuple[int, int]]]:
    """A marker whose chunk lacks the number in its text, moved to the one
    chunk of the turn that has it; the (old, new) pairs moved. The answer
    said Onur Demirörs' office was 107 and cited his PhD student's profile
    (office 138), which names him as supervisor: the judge, seeing only the
    cited chunk, failed a right answer. Only numbers decide: a name is in
    both profiles. When no chunk or several hold the number, nothing moves."""
    moved = []
    lines = []
    for line in answer.split("\n"):
        out, start = [], 0
        for run in _RUN.finditer(line):
            text = line[start:run.start()]
            cited = [int(n) for n in _MARKER.findall(run.group(2))]
            marker = run.group(0)
            if len(cited) == 1 and 1 <= cited[0] <= len(chunks):
                n = cited[0]
                # 1-2 digit numbers ("6", "29") turn up in many chunks by chance
                missing = {x for x in _NUMBER.findall(text)
                           if len(x) >= 3 and not _holds(chunks[n - 1], x)}
                holders = [m for m in range(1, len(chunks) + 1) if m != n
                           and missing and all(_holds(chunks[m - 1], x) for x in missing)]
                if len(holders) == 1:
                    moved.append((n, holders[0]))
                    marker = f"{run.group(1)}[{holders[0]}]"
            out.append(text + marker)
            start = run.end()
        out.append(line[start:])
        lines.append("".join(out))
    return "\n".join(lines), moved
