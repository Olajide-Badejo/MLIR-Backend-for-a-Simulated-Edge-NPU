# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The dash linter.

Ground rule 3 of the build specification bans em dashes and en dashes
everywhere, and bans the TeX ligature spellings ``--`` and ``---`` in ``.tex``
prose. It grants exactly one exemption, and the exemption is the reason this is
a real program rather than a grep: a command line flag such as ``--check`` is
not prose, so verbatim, lstlisting and minted environments and ``\\texttt{}``
spans in ``.tex``, and fenced code blocks and backticked spans in markdown, may
contain the double hyphen. The exemption covers the double hyphen only. U+2014
and U+2013 are rejected inside code blocks too, because no command line ever
needs one.

Encoding the exemption here rather than remembering it is the point. A linter
that everyone has to remember to override is a linter that gets overridden.

**This module runs on whatever `python3` is nearest and it may not be a recent
one.** ``dash-lint.sh`` prefers the project venv and falls back to `python3` on
`PATH`, deliberately, because CI calls this before any venv exists and the
linter has no third party dependencies to need one. So the interpreter here is
the *lowest* in the project, not the highest: it is Ubuntu 24.04's 3.12 inside
the CI container and 3.14 in the venv. `requires-python` in `pyproject.toml`
says 3.11 and this file is held to it.

That is not a style preference, it is D-0038: two unparenthesised `except A, B:`
clauses, which PEP 758 made legal in 3.14 and which are a `SyntaxError`
everywhere below it, made the whole linter fail to parse in the container the
first time anything ran it there. `[tool.ruff] target-version` is now `py311`
rather than `py314`, so the tool catches the next one instead of a CI run three
phases later.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

EM_DASH = "\u2014"
EN_DASH = "\u2013"

# Extensions worth scanning. Everything else is either binary or generated.
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".td",
    ".mlir",
    ".py",
    ".sh",
    ".bash",
    ".md",
    ".tex",
    ".bib",
    ".rst",
    ".txt",
    ".cmake",
    ".in",
    ".yml",
    ".yaml",
    ".toml",
    ".cfg",
    ".json",
}
# Files without a suffix that are still worth scanning.
TEXT_NAMES = {"CMakeLists.txt", "Makefile", "Dockerfile", "LICENSE"}

# Directories never scanned: build output, caches, and the licence texts, which
# are verbatim third party documents that this project may not rewrite.
SKIP_DIRS = {
    ".git",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    "LICENSES",
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    col: int
    rule: str
    text: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.rule}: {self.text}"


def _mask_spans(line: str, spans: list[tuple[int, int]]) -> str:
    """Blank out the given half open spans, keeping the line length."""
    chars = list(line)
    for start, end in spans:
        for i in range(max(0, start), min(len(chars), end)):
            chars[i] = " "
    return "".join(chars)


_TEX_TEXTTT = re.compile(r"\\(?:texttt|verb|url|href|path|lstinline)\b")
_TEX_VERB_BEGIN = re.compile(
    r"\\begin\{(verbatim|lstlisting|minted|Verbatim|alltt)\*?\}"
)
_TEX_VERB_END = re.compile(r"\\end\{(verbatim|lstlisting|minted|Verbatim|alltt)\*?\}")


def _brace_span(line: str, open_idx: int) -> tuple[int, int]:
    """Return the half open span of a balanced {...} group starting at or after
    open_idx, or an empty span if the group does not close on this line."""
    i = line.find("{", open_idx)
    if i < 0:
        return (0, 0)
    depth = 0
    for j in range(i, len(line)):
        if line[j] == "{":
            depth += 1
        elif line[j] == "}":
            depth -= 1
            if depth == 0:
                return (i, j + 1)
    # Unbalanced on this line: exempt to end of line rather than guess.
    return (i, len(line))


def _tex_exempt_spans(line: str) -> list[tuple[int, int]]:
    """Spans on a .tex line where a double hyphen is allowed."""
    spans: list[tuple[int, int]] = []
    for m in _TEX_TEXTTT.finditer(line):
        spans.append(_brace_span(line, m.end()))
    return [s for s in spans if s[1] > s[0]]


def _find_all(line: str, needle: str) -> list[int]:
    out: list[int] = []
    start = 0
    while True:
        i = line.find(needle, start)
        if i < 0:
            return out
        out.append(i)
        start = i + 1


def check_text(path: str, text: str) -> list[Violation]:
    """Check one file's contents and return every violation in it."""
    suffix = Path(path).suffix.lower()
    is_tex = suffix in {".tex", ".sty", ".cls"}

    violations: list[Violation] = []
    in_tex_verbatim = False

    for lineno, raw in enumerate(text.splitlines(), start=1):
        # Unicode dashes are rejected everywhere, including inside code blocks
        # and verbatim environments, with no exemption of any kind. This check
        # runs before any masking for exactly that reason.
        for char, name in ((EM_DASH, "em dash U+2014"), (EN_DASH, "en dash U+2013")):
            for idx in _find_all(raw, char):
                violations.append(
                    Violation(
                        path,
                        lineno,
                        idx + 1,
                        "unicode-dash",
                        f"{name} is banned everywhere, use an ASCII hyphen",
                    )
                )

        # From here on the line is only examined for the double hyphen, and
        # that rule exists only for .tex. The reason is typographic rather than
        # stylistic: TeX turns -- into an en dash and --- into an em dash when
        # it typesets prose, so a double hyphen that nobody wrote as a dash
        # still prints as one. No other format does that, so in markdown,
        # Python, C++ and the rest a double hyphen is just two hyphens and is
        # left alone. That is also why markdown needs no fence tracking here:
        # the exemption that fences would grant is an exemption from a rule
        # markdown never had. What markdown does keep is the unicode dash rule
        # above, which applies inside a fenced block exactly as it does
        # outside, and the fixture pins both halves of that.
        if not is_tex:
            continue

        if _TEX_VERB_END.search(raw):
            in_tex_verbatim = False
            continue
        if _TEX_VERB_BEGIN.search(raw):
            in_tex_verbatim = True
            continue
        if in_tex_verbatim:
            continue
        # A .tex comment is not typeset, so a double hyphen in one cannot make
        # a ligature. Strip from an unescaped % to end of line.
        line = raw
        for i, ch in enumerate(line):
            if ch == "%" and (i == 0 or line[i - 1] != "\\"):
                line = line[:i]
                break

        line = _mask_spans(line, _tex_exempt_spans(line))

        for idx in _find_all(line, "--"):
            # Report the run once: skip an index whose predecessor also matched.
            if idx > 0 and line[idx - 1] == "-":
                continue
            run = len(line[idx:]) - len(line[idx:].lstrip("-"))
            violations.append(
                Violation(
                    path,
                    lineno,
                    idx + 1,
                    "tex-ligature",
                    f"{'-' * run} in .tex prose becomes a dash ligature; "
                    f"use an ASCII hyphen, or wrap it in \\texttt{{}} or "
                    f"a verbatim environment if it is a flag",
                )
            )

    return violations


def is_text_file(p: Path) -> bool:
    if p.name in TEXT_NAMES:
        return True
    return p.suffix.lower() in TEXT_SUFFIXES


def iter_repo_files(root: Path, explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(e) for e in explicit]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
        ).stdout.decode()
        names = [n for n in out.split("\0") if n]
        return [root / n for n in names]
    except (subprocess.CalledProcessError, FileNotFoundError):
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                found.append(Path(dirpath) / f)
        return found


def lint_paths(root: Path, explicit: list[str]) -> list[Violation]:
    violations: list[Violation] = []
    for p in iter_repo_files(root, explicit):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        if not is_text_file(p):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = os.path.relpath(p, root)
        violations.extend(check_text(rel, text))
    return violations


def run_self_test(fixture_dir: Path) -> int:
    """Prove the exemption in both directions against the shipped fixture.

    Each fixture file is named for what it must do. A file whose name contains
    `pass` must produce no violations; one whose name contains `fail` must
    produce at least one, and must produce it for the reason the name gives.
    Asserting the reason matters: a fixture that fails for the wrong rule is a
    linter that is right by accident.
    """
    expectations = [
        ("md-flag-in-fence-pass.md", None),
        ("md-emdash-in-fence-fail.md", "unicode-dash"),
        ("tex-flag-in-verbatim-pass.tex", None),
        ("tex-emdash-in-verbatim-fail.tex", "unicode-dash"),
        ("tex-flag-in-texttt-pass.tex", None),
        ("tex-doublehyphen-in-prose-fail.tex", "tex-ligature"),
        ("py-flag-in-comment-pass.py", None),
        ("py-endash-fail.py", "unicode-dash"),
    ]
    failures = 0
    for name, expected_rule in expectations:
        path = fixture_dir / name
        if not path.is_file():
            print(f"self-test: MISSING fixture {path}", file=sys.stderr)
            failures += 1
            continue
        got = check_text(name, path.read_text(encoding="utf-8"))
        if expected_rule is None:
            if got:
                print(
                    f"self-test: FAIL {name} should be clean but reported:",
                    file=sys.stderr,
                )
                for v in got:
                    print(f"  {v}", file=sys.stderr)
                failures += 1
            else:
                print(f"self-test: ok   {name} (clean, as required)")
        else:
            rules = {v.rule for v in got}
            if expected_rule not in rules:
                print(
                    f"self-test: FAIL {name} should report {expected_rule} "
                    f"but reported {sorted(rules) or 'nothing'}",
                    file=sys.stderr,
                )
                failures += 1
            else:
                print(f"self-test: ok   {name} (reports {expected_rule}, as required)")

    if failures:
        print(f"self-test: {failures} expectation(s) not met", file=sys.stderr)
        return 1
    print(f"self-test: all {len(expectations)} expectations met")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="dash-lint",
        description="Reject em dashes and en dashes everywhere, and TeX dash "
        "ligatures in .tex prose.",
    )
    ap.add_argument(
        "paths",
        nargs="*",
        help="files to check; default is every tracked file in the repository",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="check the linter against its own fixture and exit",
    )
    ap.add_argument("--root", default=None, help="repository root")
    args = ap.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else script_dir.parent

    if args.self_test:
        return run_self_test(script_dir / "testdata" / "dash-lint-fixture")

    # The fixture is deliberately full of violations, so it is never linted as
    # part of the repository sweep. --self-test is how it is checked, and
    # REUSE.toml is how it gets its SPDX tags, because a header in a fixture
    # file would change the content the fixture exists to pin.
    fixture = (script_dir / "testdata" / "dash-lint-fixture").resolve()
    violations = [
        v
        for v in lint_paths(root, args.paths)
        if not str((root / v.path).resolve()).startswith(str(fixture))
    ]

    for v in violations:
        print(v)
    if violations:
        print(f"\ndash-lint: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    print("dash-lint: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
