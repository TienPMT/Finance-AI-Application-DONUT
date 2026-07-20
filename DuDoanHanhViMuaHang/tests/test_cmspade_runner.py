from pathlib import Path

import pytest

from mining.run_cmspade import (
    CmSpadeRunError,
    build_java_command,
    parse_pattern_count,
    parse_stdout_count,
    run_cmspade,
    validate_minsup,
)


def test_wrapper_rejects_missing_input_path(tmp_path: Path) -> None:
    output_path = tmp_path / "patterns.txt"

    with pytest.raises(FileNotFoundError, match="sequence DB not found"):
        run_cmspade(tmp_path / "missing.txt", output_path, 0.5)


@pytest.mark.parametrize("minsup", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_wrapper_rejects_invalid_minsup(minsup: float) -> None:
    with pytest.raises(ValueError, match="minsup must be in"):
        validate_minsup(minsup)


def test_command_builder_uses_provided_paths_not_sample_resource(tmp_path: Path) -> None:
    input_path = tmp_path / "sequence-db.txt"
    output_path = tmp_path / "patterns-raw.txt"
    input_path.write_text("1 -1 -2\n", encoding="utf-8")

    command = build_java_command(input_path, output_path, 0.5, verbose=True)

    assert str(input_path.resolve()) in command
    assert str(output_path.resolve()) in command
    assert "contextPrefixSpan.txt" not in command
    assert "--verbose" in command


def test_fixture_run_creates_raw_patterns(tmp_path: Path) -> None:
    input_path = Path("tests/fixtures/sequence-db-small.txt")
    output_path = tmp_path / "patterns-raw.txt"

    result = run_cmspade(input_path, output_path, 0.5)

    assert output_path.exists()
    assert "#SUP:" in output_path.read_text(encoding="utf-8")
    assert result.sequence_count == 4
    assert result.pattern_count > 0
    assert result.output_path == output_path.resolve()


def test_stdout_count_parsing_and_pattern_count_fallback(tmp_path: Path) -> None:
    output_path = tmp_path / "patterns-raw.txt"
    output_path.write_text("1 -1 #SUP: 3\n\n2 -1 #SUP: 2\n", encoding="utf-8")

    assert parse_stdout_count("sequence_count=4\npattern_count=2\n", "sequence_count") == 4
    assert parse_stdout_count("pattern_count=2\n", "sequence_count") is None
    assert parse_pattern_count("", output_path) == 2


def test_fixture_run_allows_blank_and_comment_lines(tmp_path: Path) -> None:
    input_path = tmp_path / "sequence-db-with-comments.txt"
    output_path = tmp_path / "patterns-raw.txt"
    input_path.write_text(
        "# generated fixture\n\n1 -1 2 -1 -2\n% metadata\n1 -1 2 -1 -2\n",
        encoding="utf-8",
    )

    result = run_cmspade(input_path, output_path, 0.5)

    assert "#SUP:" in output_path.read_text(encoding="utf-8")
    assert result.pattern_count > 0


@pytest.mark.parametrize(
    "bad_line",
    [
        "1 -1 2 -1",
        "1 -2 2",
        "1 2 -2",
        "1 -1 -1 -2",
    ],
)
def test_process_rejects_malformed_sequence_grammar(tmp_path: Path, bad_line: str) -> None:
    input_path = tmp_path / "bad-sequence-db.txt"
    output_path = tmp_path / "patterns-raw.txt"
    input_path.write_text(f"1 -1 2 -1 -2\n{bad_line}\n", encoding="utf-8")

    with pytest.raises(CmSpadeRunError) as error:
        run_cmspade(input_path, output_path, 0.5)

    assert "invalid sequence grammar" in str(error.value)


def test_process_failure_surfaces_command_output(tmp_path: Path) -> None:
    input_path = tmp_path / "bad-sequence-db.txt"
    output_path = tmp_path / "patterns-raw.txt"
    input_path.write_text("not-an-integer -1 -2\n", encoding="utf-8")

    with pytest.raises(CmSpadeRunError) as error:
        run_cmspade(input_path, output_path, 0.5)

    message = str(error.value)
    assert "CM-SPADE failed" in message
    assert "not-an-integer" in message or "NumberFormatException" in message
