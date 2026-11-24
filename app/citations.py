"""Checks the [n] markers in an answer against the chunks they point to.

The model cites reference chunks as [n]. Now and then it numbers them like
paper footnotes instead: the facts of chunks [36]-[38] come out as [1]-[3]
with a "[1] [2] [3]" line at the end, and the UI shows chips for whatever
chunks [1]-[3] were. Checks that catch it without another model call:
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
