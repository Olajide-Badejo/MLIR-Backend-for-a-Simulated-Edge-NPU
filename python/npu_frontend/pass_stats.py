# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""The reader for Section 16.2's instrumentation, and the check that it is whole.

`lib/Pipeline/PassStats.cpp` writes the file. This reads it, and the reading is
where the gate clause lives: **a pass present in the pipeline but absent from
the JSON raises.**

**Why the check is here and not in the C++.** The instrumentation knows which
passes ran. It does not know which passes were *meant* to run, because that is a
property of the level the driver asked for, of the stage it asked to stop at, and
of the ablation it asked for, and all three are the driver's. The instrumentation
reporting on itself would be a check with one subject. So the file records what
happened, this module holds the expectation, and the two are compared by
something that saw both.

**The expectation is read from the compiler at run time.** `expected_passes`
goes through `describe_pipeline`, which is `npu-opt --npu-describe-pipeline`,
which is `lib/Pipeline/Pipeline.cpp`'s own table. Section 16.2 requires that of
the ablatable set and the same argument applies to the whole list: a copy written
here would stop covering a pass on the day one was added, and nothing would go
red.

**The cross check against `--mlir-timing` has a direction, and the direction is
the evidence.** Both clocks measure the same run of the same pipeline. MLIR's
timer is started before this project's instrumentation is called and stopped
after it, so MLIR's figure per pass contains the instrumentation's operation
walk and this project's does not, and MLIR's is therefore always the larger. That
is not a caveat to work around; it is what makes the comparison a check rather
than a tautology, because a per pass figure that came out *smaller* than MLIR's
by more than the display resolution, or larger than it at all, would mean the two
were not measuring the same thing.
"""

from __future__ import annotations

import json
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .compile import describe_pipeline

#: Every key `lib/Pipeline/PassStats.cpp` writes for one pass. A file missing
#: one of them is a file from a different version of the instrumentation, and
#: that is a refusal rather than a default: Section 16.2 says a missing timing
#: or op count is an error and never a zero.
REQUIRED_PASS_KEYS: Final[tuple[str, ...]] = (
    "position",
    "name",
    "pass_name",
    "anchor_op",
    "ops_before",
    "ops_after",
    "ops_before_total",
    "ops_after_total",
    "wall_ms",
    "pass_timing_source",
)

#: The keys of the file itself.
REQUIRED_ROOT_KEYS: Final[tuple[str, ...]] = (
    "generated_by",
    "source",
    "clock",
    "run_completed",
    "passes_total_wall_ms",
    "passes",
    "still_running_at_exit",
    "unmatched_after_pass",
)

#: The fraction of a pass's time this instrumentation's own operation walk may
#: add to MLIR's figure.
#:
#: The walk sits inside MLIR's window and outside this project's, and it is work
#: proportional to the module and therefore to the time the pass took. A fixed
#: allowance would be one that held on this machine and went red on a slower one
#: for a reason that is not a defect, which is the mistake `tolerances.py`
#: records having been made once already about `onnxruntime`.
#:
#: **Measured on 2026-09-01** over all seven models at all three levels and both
#: batch sizes: the worst gap was 0.069 ms on a pass MLIR timed at 1.1 ms, which
#: is 6 percent. Half is a wide allowance against that. Never widened to make a
#: cell pass: a pass outside it is a finding about the two clocks.
TIMING_GAP_FRACTION: Final[float] = 0.5

#: How many decimals of seconds MLIR's tree display prints, as this module was
#: written against it.
#:
#: **Not used as a constant in any comparison**, which is D-0043. It is here so
#: the parser can say when what it read stops matching what this was written
#: against. Every bound derives its quantum from the text actually parsed, per
#: report, so that a format change moves the bound instead of silently
#: invalidating it.
MLIR_TIMING_DECIMALS: Final[int] = 4


class PassStatisticsError(Exception):
    """The recorded pass statistics are not the ones the pipeline ran."""


@dataclass(frozen=True)
class PassRecord:
    """One pass invocation, as the instrumentation recorded it."""

    position: int
    name: str
    pass_name: str
    anchor_op: str
    ops_before: dict[str, int]
    ops_after: dict[str, int]
    ops_before_total: int
    ops_after_total: int
    wall_ms: float
    pass_timing_source: str

    @property
    def ops_delta(self) -> int:
        """What the pass did to the operation count. Negative is a removal."""
        return self.ops_after_total - self.ops_before_total


# ---------------------------------------------------------------------------
# What the pipeline was meant to run.
# ---------------------------------------------------------------------------


def expected_passes(
    level: int, *, stage: str = "npuisa", ablated: str | None = None
) -> list[str]:
    """The pass arguments a level runs, in order, as the compiler describes it.

    `stage` is `npu` for the tensor level half, which is what
    ``npu-compile --emit npu`` runs, and `npuisa` for the whole level.

    `ablated` removes every entry with that argument, which is both entries when
    the argument is `canonicalize` at `-O2`. That mirrors `build()` in
    `lib/Pipeline/Pipeline.cpp`, and it mirrors it by reading the same two
    fields out of the same table rather than by reimplementing the decision.
    """
    if stage not in ("npu", "npuisa"):
        raise PassStatisticsError(
            f"{stage!r} is not a pipeline stage. The stages a pipeline stops "
            f"after are 'npu' and 'npuisa'."
        )

    rows = None
    for row in describe_pipeline()["levels"]:
        if int(row["level"]) == level:
            rows = row["passes"]
            break
    if rows is None:
        raise PassStatisticsError(f"-O{level} is not an optimization level")

    names: list[str] = []
    for entry in rows:
        if stage == "npu" and entry["stage"] != "npu":
            continue
        if ablated is not None and entry["ablatable"] and entry["pass"] == ablated:
            continue
        names.append(str(entry["pass"]))
    return names


# ---------------------------------------------------------------------------
# Reading the file, and refusing a file that does not match.
# ---------------------------------------------------------------------------


def load_pass_stats(
    path: str | Path, *, expected: list[str] | None = None
) -> list[PassRecord]:
    """The pass records, or a refusal that names what is wrong.

    Four refusals, and each is a different fault:

    - the file is missing a key, which means it came from a different build of
      the instrumentation than this reader was written against;
    - the run did not complete, which means a pass failed and the counts after
      it describe an operation that was left half rewritten;
    - a pass the pipeline was meant to run is absent, which is the gate clause;
    - a pass ran that the pipeline was not meant to run, which is the same fault
      seen from the other side and is what an ablation that quietly did nothing
      looks like.
    """
    source = Path(path)
    if not source.is_file():
        raise PassStatisticsError(
            f"{source} was not written. The instrumentation writes it when the "
            f"pass manager is destroyed, so an absent file means the compiler "
            f"never got as far as building a pipeline."
        )

    document: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    missing_root = [key for key in REQUIRED_ROOT_KEYS if key not in document]
    if missing_root:
        raise PassStatisticsError(
            f"{source} is missing {missing_root}. Section 16.2 makes a missing "
            f"field an error and never a zero, so this is refused rather than "
            f"defaulted."
        )

    if not document["run_completed"]:
        raise PassStatisticsError(
            f"{source} records a run that did not complete: the pipeline failed "
            f"at {document.get('failed_pass')!r}. The counts after a failed pass "
            f"describe an operation the pass may have left half rewritten, so "
            f"they are not a measurement of anything."
        )
    if document["still_running_at_exit"] or document["unmatched_after_pass"]:
        raise PassStatisticsError(
            f"{source} records passes the instrumentation never closed "
            f"({document['still_running_at_exit']}) or closed without opening "
            f"({document['unmatched_after_pass']}). Either is a pass manager "
            f"doing something this instrumentation was not written for."
        )

    records: list[PassRecord] = []
    for index, entry in enumerate(document["passes"]):
        absent = [key for key in REQUIRED_PASS_KEYS if key not in entry]
        if absent:
            raise PassStatisticsError(
                f"{source}: the pass at position {index} is missing {absent}."
            )
        records.append(
            PassRecord(
                position=int(entry["position"]),
                name=str(entry["name"]),
                pass_name=str(entry["pass_name"]),
                anchor_op=str(entry["anchor_op"]),
                ops_before={str(k): int(v) for k, v in entry["ops_before"].items()},
                ops_after={str(k): int(v) for k, v in entry["ops_after"].items()},
                ops_before_total=int(entry["ops_before_total"]),
                ops_after_total=int(entry["ops_after_total"]),
                wall_ms=float(entry["wall_ms"]),
                pass_timing_source=str(entry["pass_timing_source"]),
            )
        )

    if expected is not None:
        _assert_covers(records, expected, source)
    return records


def _assert_covers(
    records: list[PassRecord], expected: list[str], source: Path
) -> None:
    """The gate clause: a pass in the pipeline and not in the JSON raises."""
    recorded = [record.name for record in records]
    if recorded == expected:
        return

    absent = _multiset_difference(expected, recorded)
    extra = _multiset_difference(recorded, expected)
    lines = [
        f"{source} does not describe the pipeline that was asked for.",
        f"  the pipeline runs: {expected}",
        f"  the file records:  {recorded}",
    ]
    if absent:
        lines.append(
            f"  in the pipeline and absent from the file: {absent}. Section "
            f"16.2 makes that a refusal: a pass whose before and after counts "
            f"are missing is a pass whose ablation row would be a subtraction "
            f"nobody could check."
        )
    if extra:
        lines.append(
            f"  recorded and not in the pipeline: {extra}. On an ablation run "
            f"this is what a removal that did not happen looks like, and it is "
            f"the reason the removal is checked by measurement rather than "
            f"trusted."
        )
    if not absent and not extra:
        lines.append(
            "  the same passes ran in a different order, which is a pipeline "
            "that is not the one the table describes."
        )
    raise PassStatisticsError("\n".join(lines))


def _multiset_difference(left: list[str], right: list[str]) -> list[str]:
    remaining = list(right)
    difference: list[str] = []
    for name in left:
        if name in remaining:
            remaining.remove(name)
        else:
            difference.append(name)
    return difference


# ---------------------------------------------------------------------------
# The independent cross check.
# ---------------------------------------------------------------------------

#: One row of `--mlir-timing --mlir-timing-display=tree`. The leading columns
#: are the wall time in seconds and the percentage; the name is the rest.
_TIMING_ROW = re.compile(r"^\s*([0-9]+\.[0-9]+)\s*\(\s*[0-9.]+%\)\s+(\S.*?)\s*$")

#: The rows that are not passes: the parser, the printer, the residual, the
#: total, the pipeline group headers, and the analyses MLIR nests under a pass.
_NOT_A_PASS = ("Parser", "Output", "Rest", "Total", "root")


@dataclass(frozen=True)
class TimingReport:
    """What one `--mlir-timing` tree display said, and how precisely it said it.

    **The precision is part of the reading**, which is D-0043. These figures are
    printed to a fixed number of decimals, so each one is a rounding of a number
    this project cannot see, and every comparison against them has to carry the
    quantum of that rounding. Returning the rows without it invited exactly the
    assertion that went red: a strict `>=` between a sum of rounded values and a
    sum of exact ones.
    """

    rows: list[tuple[str, float]]
    #: Half a unit in the last place of the printed figures, in milliseconds.
    #: Derived from the text that was parsed rather than assumed, so a format
    #: change moves the bound rather than silently invalidating it.
    half_ulp_ms: float
    #: How many decimals of seconds this report printed.
    decimals: int


def interpreter_is_traced() -> bool:
    """Whether something is tracing or monitoring this interpreter.

    *Added at P11, with the coverage measurement over `experiments/`.* A tracer
    stretches whatever runs between two C++ timestamps, and one of this module's
    bounds is about exactly that interval, so the bound has to know.

    Both mechanisms are checked because CPython has two and `coverage` picks per
    version: `sys.settrace` on older interpreters, and `sys.monitoring` from
    3.12. `sys.gettrace()` alone reports `None` under the monitoring backend,
    which would make this answer "not traced" on the interpreter this project
    actually runs, so a check that only asked it would be wrong in exactly the
    environment it exists for.

    `threading.gettrace()` covers a tracer installed for new threads but not for
    this one, which is how a test harness that starts the work on a worker thread
    would look.
    """
    if sys.gettrace() is not None or threading.gettrace() is not None:
        return True
    monitoring = getattr(sys, "monitoring", None)
    if monitoring is None:
        return False
    # Any tool registered at all: `coverage` takes `COVERAGE_ID`, but naming one
    # id would miss a profiler or a debugger doing the same thing for the same
    # reason.
    for tool_id in range(getattr(monitoring, "_MAX_TOOLS", 6)):
        try:
            if monitoring.get_tool(tool_id) is not None:
                return True
        except (ValueError, AttributeError):
            continue
    return False


def parse_mlir_timing(text: str) -> TimingReport:
    """The per pass rows of a tree display, in order, with their print precision.

    Nothing else in this project parses a tool's human readable output, and the
    exception is deliberate rather than an oversight: this is the *independent*
    half of the cross check, and the whole point of it is that it comes from
    MLIR's own timing machinery rather than from the instrumentation being
    checked. Reading it through a second channel this project also wrote would
    make the two agree for the wrong reason.

    **The price of reading printed output is that every value is a rounding**,
    and this returns the size of that rounding beside the values. MLIR prints
    seconds to four decimals, so every figure is a multiple of 0.1 ms and is
    within 0.05 ms of the truth. Measured, on the eleven passes of `-O2`: the
    parsed values are 0.1, 0.2, 0.1, 0.1, 0.4, 0.4, 0.2, 0.3, 0.2, 0.5, 0.2.
    """
    rows: list[tuple[str, float]] = []
    decimals = 0
    for line in text.splitlines():
        match = _TIMING_ROW.match(line)
        if not match:
            continue
        name = match.group(2)
        if name in _NOT_A_PASS or name.startswith("(A)") or "Pipeline" in name:
            continue
        seconds = match.group(1)
        # The coarsest row decides, because a sum is as coarse as its coarsest
        # term. Every row of one report has the same format today; taking the
        # maximum means a report that ever mixed them stays sound.
        decimals = max(decimals, len(seconds.split(".")[1]))
        rows.append((name, float(seconds) * 1000.0))

    # Seconds printed to `decimals` places is a quantum of 10**-decimals
    # seconds, which is 10**(3 - decimals) milliseconds, and a rounded figure is
    # within half of that of the truth.
    half_ulp = 10.0 ** (3 - decimals) / 2.0 if rows else 0.0
    return TimingReport(rows=rows, half_ulp_ms=half_ulp, decimals=decimals)


@dataclass(frozen=True)
class CrossCheck:
    """What the two clocks said about the same run."""

    #: (pass name, instrumented ms, mlir ms, mlir minus instrumented).
    rows: list[tuple[str, float, float, float]]
    instrumented_total_ms: float
    mlir_total_ms: float
    #: Half a unit in the last place of MLIR's printed figures, from the report
    #: this check read. Every bound here is expressed in terms of it.
    half_ulp_ms: float

    #: Empty when every bound ran. Otherwise why the per pass gap bound did not,
    #: which today has exactly one cause: a traced interpreter.
    #:
    #: **A string rather than a boolean, and set rather than defaulted**, because
    #: Section 19.0's rule is that a check which did not run says so. A caller
    #: that prints this cannot present a skipped bound as a passed one, and a
    #: caller that ignores it is visibly ignoring something.
    upper_bound_skipped: str = ""

    @property
    def worst_gap_ms(self) -> float:
        return max((row[3] for row in self.rows), default=0.0)

    @property
    def worst_gap_fraction(self) -> float:
        """The worst gap as a fraction of what MLIR timed the pass at."""
        return max(
            (row[3] / row[2] for row in self.rows if row[2] > 0.0),
            default=0.0,
        )

    @property
    def worst_deficit_ms(self) -> float:
        """How far MLIR ever came out *below* the instrumentation."""
        return max((-row[3] for row in self.rows), default=0.0)

    @property
    def rounding_allowance_ms(self) -> float:
        """How far the sum of MLIR's printed figures may fall below the truth.

        **This is D-0043.** Each printed figure is within half a unit in the last
        place of the value it stands for, so a sum of `n` of them is within `n`
        times that of the true sum. Comparing the two totals with a strict `>=`
        ignored the quantum of the coarser one entirely, and went red in CI on a
        1.7 microsecond margin between a one decimal number and a microsecond
        resolution one.

        Scaling with the pass count rather than being one number is the point: a
        level with more passes accumulates more rounding, and a bound that did
        not know how many rows it summed would be right at one level and wrong at
        another.
        """
        return len(self.rows) * self.half_ulp_ms

    @property
    def totals_agree(self) -> bool:
        """Whether MLIR's total is at or above this project's, allowing rounding.

        The direction is the substantive claim and it survives: MLIR's timer
        opens before this instrumentation is called and closes after it, so its
        figure contains the operation walk and this project's does not. What the
        allowance concedes is only that the containment is observed through
        figures printed to a finite number of decimals.
        """
        return (
            self.mlir_total_ms
            >= self.instrumented_total_ms - self.rounding_allowance_ms
        )


def cross_check_against_mlir_timing(
    records: list[PassRecord], timing_text: str
) -> CrossCheck:
    """Compares the instrumentation's clock against MLIR's, and refuses a gap.

    The comparison is by `pass_name`, which is `Pass::getName()` and is exactly
    the string MLIR's timing prints. The argument is what this project keys its
    table on and MLIR has never heard of it, so matching on the argument would
    need a translation table and a translation table is a third place for the
    two to disagree.
    """
    report = parse_mlir_timing(timing_text)
    mlir_rows = report.rows
    ours = [(record.pass_name, record.wall_ms) for record in records]

    if [name for name, _ in mlir_rows] != [name for name, _ in ours]:
        raise PassStatisticsError(
            "the instrumentation and --mlir-timing disagree about which passes "
            "ran, which makes any comparison of their clocks meaningless.\n"
            f"  instrumentation: {[name for name, _ in ours]}\n"
            f"  --mlir-timing:   {[name for name, _ in mlir_rows]}"
        )

    rows: list[tuple[str, float, float, float]] = []
    for (name, instrumented), (_, from_mlir) in zip(ours, mlir_rows, strict=True):
        rows.append((name, instrumented, from_mlir, from_mlir - instrumented))

    check = CrossCheck(
        rows=rows,
        instrumented_total_ms=sum(row[1] for row in rows),
        mlir_total_ms=sum(row[2] for row in rows),
        half_ulp_ms=report.half_ulp_ms,
        upper_bound_skipped=(
            (
                "the interpreter is traced, so the window MLIR times holds the "
                "tracer as well as this instrumentation's operation walk, and "
                "the gap bound's premise that the gap is the walk does not "
                "hold. The deficit bound and the totals were still checked."
            )
            if interpreter_is_traced()
            else ""
        ),
    )

    # The deficit bound is exactly the print quantum and is derived rather than
    # chosen, which is D-0043. MLIR's true figure contains this project's span,
    # so `true_mlir >= instrumented`; the printed figure is within half a unit in
    # the last place of the true one, so `printed >= instrumented - half_ulp`.
    # Any deficit larger than that is the two clocks not measuring the same run.
    if check.worst_deficit_ms > report.half_ulp_ms:
        offender = min(rows, key=lambda row: row[3])
        raise PassStatisticsError(
            f"--mlir-timing reports {offender[0]} at {offender[2]:.4f} ms and "
            f"this project's instrumentation at {offender[1]:.4f} ms, which is "
            f"{-offender[3]:.4f} ms below it against a bound of "
            f"{report.half_ulp_ms:.4f} ms. MLIR's timer opens before this "
            f"instrumentation is called and closes after it, so its figure "
            f"contains ours and can only come out below it by the rounding of "
            f"its own display, which prints seconds to {report.decimals} "
            f"decimals. A deficit larger than that means the two are not "
            f"measuring the same run."
        )
    # The upper bound has the same two terms for the same reason: the print
    # quantum, because a rounded figure can sit half a unit *above* the truth as
    # readily as below it, plus the walk, which is proportional to the pass.
    #
    # **It is not checked under a traced interpreter, and that is a precondition
    # rather than an exemption.** This bound's premise, stated in its own
    # message, is that the gap *is* the instrumentation's operation walk: the one
    # thing that sits inside MLIR's window and outside this project's. Under a
    # tracer everything else in that window is stretched too, so the gap becomes
    # the walk plus whatever the tracer did, and the bound stops measuring what
    # it says.
    #
    # Measured on 2026-09-03, the same cells three times each: untraced worst
    # gaps 0.0658, 0.0729 and 0.0690 ms; under `coverage` 0.2757, 0.0844 and
    # 0.0769. The tracer does not shift the gap, it produces occasional
    # outliers, which makes the check flaky rather than wrong, and a flaky check
    # is worth less than one that says it did not run.
    #
    # The **deficit** bound above stays active, because its premise survives a
    # tracer: MLIR's window contains this project's whatever the interpreter is
    # doing in between.
    for name, instrumented, from_mlir, gap in rows:
        if check.upper_bound_skipped:
            break
        allowed = report.half_ulp_ms + TIMING_GAP_FRACTION * from_mlir
        if gap > allowed:
            raise PassStatisticsError(
                f"--mlir-timing reports {name} at {from_mlir:.4f} ms and this "
                f"project's instrumentation at {instrumented:.4f} ms, a gap of "
                f"{gap:.4f} ms against a bound of {allowed:.4f} ms, which is "
                f"{report.half_ulp_ms:.4f} ms of display rounding plus "
                f"{TIMING_GAP_FRACTION:.0%} of MLIR's figure. The gap is this "
                f"instrumentation's own operation walk, which sits inside "
                f"MLIR's window and outside ours, so a gap this size means the "
                f"instrumentation is costing more than the pass it measures."
            )

    # The totals, which is the comparison that went red in CI. A sum of `n`
    # rounded figures is within `n` half units of the true sum, and comparing it
    # against an exact sum with no allowance was asserting a precision the
    # printed values do not have.
    if not check.totals_agree:
        raise PassStatisticsError(
            f"--mlir-timing totals {check.mlir_total_ms:.6f} ms over "
            f"{len(rows)} passes and this project's instrumentation totals "
            f"{check.instrumented_total_ms:.6f} ms, which is "
            f"{check.instrumented_total_ms - check.mlir_total_ms:.6f} ms more, "
            f"against an allowance of {check.rounding_allowance_ms:.4f} ms. "
            f"That allowance is {len(rows)} passes times the "
            f"{report.half_ulp_ms:.4f} ms each printed figure is rounded by, "
            f"and a shortfall beyond it is not rounding."
        )
    return check
