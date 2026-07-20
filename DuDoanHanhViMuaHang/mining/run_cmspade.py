import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SOURCE_DIR = ROOT / ".claude" / "reference" / "CMSpade" / "src"
RUNNER_SOURCE_DIR = ROOT / "cmspade" / "src"
BUILD_DIR = ROOT / "cmspade" / "build" / "classes"
RUNNER_CLASS = "CmSpadeFileRunner"


class CmSpadeRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class CmSpadeRunResult:
    input_path: Path
    output_path: Path
    minsup: float
    sequence_count: int
    pattern_count: int
    stdout: str
    stderr: str


def validate_minsup(minsup: float) -> None:
    if not math.isfinite(minsup) or minsup <= 0 or minsup > 1:
        raise ValueError("minsup must be in (0, 1]")


def build_java_command(
    input_path: Path,
    output_path: Path,
    minsup: float,
    verbose: bool = False,
) -> list[str]:
    validate_minsup(minsup)
    command = [
        "java",
        "-cp",
        str(BUILD_DIR),
        RUNNER_CLASS,
        "--input",
        str(input_path.resolve()),
        "--output",
        str(output_path.resolve()),
        "--minsup",
        str(minsup),
    ]
    if verbose:
        command.append("--verbose")
    return command


def run_cmspade(
    input_path: Path,
    output_path: Path,
    minsup: float,
    verbose: bool = False,
    timeout_seconds: int = 120,
) -> CmSpadeRunResult:
    validate_minsup(minsup)
    resolved_input = input_path.resolve()
    resolved_output = output_path.resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"sequence DB not found: {resolved_input}")
    ensure_java_classes()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    command = build_java_command(resolved_input, resolved_output, minsup, verbose=verbose)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise CmSpadeRunError(
            f"CM-SPADE timed out after {timeout_seconds} seconds\n"
            f"stdout:\n{error.stdout or ''}\nstderr:\n{error.stderr or ''}"
        ) from error
    if completed.returncode != 0:
        raise CmSpadeRunError(
            "CM-SPADE failed with exit code "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not resolved_output.exists():
        raise CmSpadeRunError(
            f"CM-SPADE finished without creating output file: {resolved_output}"
        )
    return CmSpadeRunResult(
        input_path=resolved_input,
        output_path=resolved_output,
        minsup=minsup,
        sequence_count=parse_stdout_count(completed.stdout, "sequence_count"),
        pattern_count=parse_pattern_count(completed.stdout, resolved_output),
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def ensure_java_classes() -> None:
    require_executable("java")
    require_executable("javac")
    sources = java_sources()
    if not sources:
        raise FileNotFoundError("CM-SPADE Java sources not found")
    runner_class_file = BUILD_DIR / f"{RUNNER_CLASS}.class"
    if (
        runner_class_file.exists()
        and dependency_class_file_exists()
        and not classes_are_stale(runner_class_file, sources)
    ):
        return
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    command = ["javac", "-encoding", "UTF-8", "-d", str(BUILD_DIR), *map(str, sources)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise CmSpadeRunError(
            "CM-SPADE Java compile failed with exit code "
            f"{completed.returncode}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def java_sources() -> list[Path]:
    return sorted(REFERENCE_SOURCE_DIR.rglob("*.java")) + sorted(RUNNER_SOURCE_DIR.rglob("*.java"))


def classes_are_stale(class_file: Path, sources: list[Path]) -> bool:
    class_mtime = class_file.stat().st_mtime
    return any(source.stat().st_mtime > class_mtime for source in sources)


def dependency_class_file_exists() -> bool:
    return (BUILD_DIR / "algorithms" / "AlgoCMSPADE.class").exists()


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise CmSpadeRunError(f"required executable not found on PATH: {name}")


def parse_pattern_count(stdout: str, output_path: Path) -> int:
    pattern_count = parse_stdout_count(stdout, "pattern_count")
    if pattern_count is not None:
        return pattern_count
    return sum(1 for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip())


def parse_stdout_count(stdout: str, key: str) -> int | None:
    for line in stdout.splitlines():
        if line.startswith(f"{key}="):
            return int(line.split("=", 1)[1])
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Java CM-SPADE on a sequence DB file.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/patterns/patterns-raw.txt"), type=Path)
    parser.add_argument("--minsup", required=True, type=float)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_cmspade(
        args.input,
        args.output,
        args.minsup,
        verbose=args.verbose,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"patterns_raw: {result.output_path}")
    print(f"pattern_count: {result.pattern_count}")


if __name__ == "__main__":
    main()
