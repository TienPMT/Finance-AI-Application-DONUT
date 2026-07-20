import pytest

from mining.parse_patterns import (
    filter_recommendable_patterns,
    parse_pattern_line,
    parse_patterns_text,
)


def test_parser_reads_sequential_pattern_with_support() -> None:
    pattern = parse_pattern_line("1 -1 6 -1 #SUP: 3")

    assert pattern.itemsets == ((1,), (6,))
    assert pattern.support == 3
    assert pattern.antecedent == (1,)
    assert pattern.consequent == (6,)
    assert pattern.raw_pattern == "1 -1 6 -1"


def test_parser_flattens_prior_itemsets_as_antecedent_for_length_three_patterns() -> None:
    pattern = parse_pattern_line("1 -1 6 7 -1 8 -1 #SUP: 3")

    assert pattern.itemsets == ((1,), (6, 7), (8,))
    assert pattern.antecedent == (1, 6, 7)
    assert pattern.consequent == (8,)


def test_parser_preserves_same_itemset_pattern_but_filter_excludes_it() -> None:
    pattern = parse_pattern_line("6 7 -1 #SUP: 2")

    assert pattern.itemsets == ((6, 7),)
    assert list(filter_recommendable_patterns([pattern])) == []


def test_parser_rejects_invalid_lines_clearly() -> None:
    with pytest.raises(ValueError, match="missing #SUP"):
        parse_pattern_line("1 -1 2 -1")

    with pytest.raises(ValueError, match="empty itemset"):
        parse_pattern_line("1 -1 -1 #SUP: 2")


def test_parser_ignores_blank_and_comment_lines() -> None:
    patterns = parse_patterns_text("# generated\n\n1 -1 #SUP: 3\n% metadata\n")

    assert len(patterns) == 1
    assert patterns[0].itemsets == ((1,),)
