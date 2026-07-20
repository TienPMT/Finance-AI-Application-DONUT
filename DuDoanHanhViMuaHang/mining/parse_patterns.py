import argparse
import re
from pathlib import Path
from typing import Iterable, Iterator

from mining.models import ParsedPattern

SUPPORT_MARKER = "#SUP:"
SUPPORT_RE = re.compile(r"^\s*(?P<pattern>.*?)\s+#SUP:\s*(?P<support>\d+)\s*$")
COMMENT_PREFIXES = ("#", "%", "@")


def parse_pattern_line(line: str) -> ParsedPattern:
    stripped = line.strip()
    match = SUPPORT_RE.match(stripped)
    if not match:
        raise ValueError(f"Pattern line missing #SUP support marker: {line!r}")
    raw_pattern = match.group("pattern").strip()
    support = int(match.group("support"))
    tokens = raw_pattern.split()
    itemsets: list[tuple[int, ...]] = []
    current: list[int] = []
    for token in tokens:
        if token == "-1":
            if not current:
                raise ValueError(f"Pattern line contains empty itemset: {line!r}")
            itemsets.append(tuple(current))
            current = []
            continue
        if token == "-2":
            raise ValueError(f"Pattern line unexpectedly contains sequence terminator: {line!r}")
        try:
            item_id = int(token)
        except ValueError as error:
            raise ValueError(f"Pattern line contains non-integer item ID: {line!r}") from error
        if item_id <= 0:
            raise ValueError(f"Pattern line contains invalid item ID: {line!r}")
        current.append(item_id)
    if current:
        raise ValueError(f"Pattern line must end each itemset with -1: {line!r}")
    if not itemsets:
        raise ValueError(f"Pattern line contains no itemsets: {line!r}")
    return ParsedPattern(itemsets=tuple(itemsets), support=support, raw_pattern=raw_pattern)


def parse_patterns_text(text: str) -> list[ParsedPattern]:
    return list(iter_parse_pattern_lines(text.splitlines()))


def iter_parse_pattern_lines(lines: Iterable[str]) -> Iterator[ParsedPattern]:
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(COMMENT_PREFIXES):
            continue
        try:
            yield parse_pattern_line(stripped)
        except ValueError as error:
            raise ValueError(f"Invalid pattern at line {line_number}: {error}") from error


def parse_patterns_file(path: Path) -> list[ParsedPattern]:
    return parse_patterns_text(path.read_text(encoding="utf-8"))


def iter_parse_patterns_file(path: Path) -> Iterator[ParsedPattern]:
    with path.open(encoding="utf-8") as input_file:
        yield from iter_parse_pattern_lines(input_file)


def filter_recommendable_patterns(
    patterns: Iterable[ParsedPattern],
    min_support: int = 1,
    max_itemsets: int = 3,
) -> list[ParsedPattern]:
    return [
        pattern
        for pattern in patterns
        if pattern.support >= min_support
        and 2 <= len(pattern.itemsets) <= max_itemsets
        and pattern.antecedent
        and pattern.consequent
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse CM-SPADE raw pattern output.")
    parser.add_argument("--input", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = parse_patterns_file(args.input)
    usable = filter_recommendable_patterns(patterns)
    print(f"patterns: {len(patterns)}")
    print(f"recommendable_patterns: {len(usable)}")


if __name__ == "__main__":
    main()
