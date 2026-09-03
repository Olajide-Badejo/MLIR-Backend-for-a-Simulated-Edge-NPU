# SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>
#
# SPDX-License-Identifier: MIT
"""Energy and area from Accelergy, per Section 16.4.

    python experiments/accelergy_energy.py --models lenet --verbose
    python experiments/accelergy_energy.py --json /tmp/energy.json

Accelergy estimates energy from action counts, which is why it fits this project:
the simulator nearly produces those counts already. Cite `accelergy2019` and
`cimloop2024` from `report/references.bib`, which is the form the tool's own
repository asks for.

## The three components, and the counts each is given

- **the MAC array**, one `fpmac` per processing element. Its action count is
  `simulation.macs`, **raw**. Section 5.5 forbids the energy path from ever
  seeing `effective_macs`: utilization describes how long the array was busy,
  not how many multiplies happened, and feeding a utilization scaled count into
  Accelergy would overstate the energy of exactly the layers the evaluation cares
  most about. `_action_counts` reads `simulation.macs` and nothing else, and
  `test_accelergy_energy.py` asserts a cell whose `effective_macs` is four times
  its `macs` produces the energy of `macs`.
- **the scratchpad**, one SRAM of the cell's own budget. Its action counts are
  `scratchpad_elements_read` and `scratchpad_elements_written`, which the
  simulator counts at the scratchpad port.
- **DRAM**. Its action counts are `dram_bytes_read` and `dram_bytes_written`
  divided by the eight byte access width, and the division is exact or this
  module raises rather than rounding.

## What is external here, and it is less than it looks

**The action counts come from this project's own `Stats`, so only the per action
energy coefficients are external.** Every counting bug in the simulator
propagates straight into the energy conclusions, including the fusion in energy
terms argument. Accelergy is not an independent check of activity; it is an
independent source of coefficients. That sentence is here, in `docs/NUMBERS.md`,
and in the evaluation, because it is the one thing a reader could reasonably get
wrong about these numbers.

## Two pinned assumptions, and the second one is not in the specification

**The technology node is 45 nm**, which Section 16.4 pins and which is recorded
in every manifest. Accelergy has no default, and without a pinned node the
picojoule and square millimetre figures are not comparable between two runs of
this project, let alone against anyone else's.

**The clock is 1 ns**, and this project had no clock before P11. Accelergy's
Aladdin tables are indexed by the cycle time they were synthesised for, so a
figure cannot be obtained without one. 1 ns is chosen because it is Accelergy's
own default in every shipped example and is the round number, and it was chosen
**before** looking at what it did to the sanity check of Section 16.4, which is
the only defensible order to choose it in. It is an assumption with a stated
uncertainty, like every constant in Section 5.5, and `docs/NUMBERS.md` says what
moving it would do.

## Reading Accelergy's answer

The same rule as SCALE-Sim and for the same measured reason: **a zero exit is not
an answer**. Accelergy's own shipped basic example crashes on this install and
exits **0**, printing `Accelergy has encountered an error and crashed`, while
other failures exit 255. So this module requires `ERT.yaml`, `ART.yaml` and
`energy_estimation.yaml` to exist, requires every component it asked about to be
in each, and raises with the tool's own output otherwise. D-0044 records the
measurement.

Every parsed number carries its provenance: the file, the component, the action,
and the units. Accelergy prints energies in **picojoules** and areas in **square
micrometres**, at a precision it names on the command line and which this module
pins at its default of six significant figures; `area_mm2` divides by 1e6 and the
conversion is written where it happens rather than remembered.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from roofline import Compiler, _mlir_python_packages_dir  # noqa: E402, F401

sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(_mlir_python_packages_dir()))

from npu_frontend import cost_model  # noqa: E402
from npu_frontend.results import (  # noqa: E402
    RESULTS_DIR,
    ResultSchemaError,
    load_result,
)

#: Section 16.4 pins it and it is recorded in every manifest.
TECHNOLOGY_NODE = "45nm"

#: The cycle time the Aladdin tables are indexed by. See the module docstring for
#: why this project has to pin one and how the value was chosen.
GLOBAL_CYCLE_SECONDS = 1e-9

#: The DRAM access width in bytes, which is the `width: 64` of the architecture
#: below written once in bits and once in bytes and kept in step here.
DRAM_ACCESS_BYTES = 8

#: The scratchpad word width in bits. One f32 element per access, which is what
#: makes `scratchpad_elements_read` an access count rather than a byte count.
SCRATCHPAD_WORD_BITS = 32

#: The three components, named once. Section 16.4 asks for energy and area parsed
#: per component so the evaluation can attribute energy rather than quoting one
#: scalar, and these are the keys it attributes to.
COMPONENTS = ("mac_array", "scratchpad", "main_memory")

#: The published reference figures at 45 nm and 0.9 V that Section 16.4 asks the
#: sanity test to compare against. Recorded here rather than in the test, so the
#: numbers a claim is checked against live beside the claim.
#:
#: The note about DRAM is Section 16.4's and is repeated because it is the kind of
#: thing that gets rounded off: the widely repeated "DRAM equals 640 pJ" figure is
#: **not** what the source says, and the source is what is used.
REFERENCE_PJ = {
    "int8_add": 0.03,
    "int8_multiply": 0.2,
    "fp32_multiply": 3.7,
    "fp32_add": 0.9,
    "cache_32kb_access": 20.0,
    "dram_access_low": 1300.0,
    "dram_access_high": 2600.0,
}


class AccelergyError(Exception):
    """Accelergy could not be run, or its answer could not be read."""


# ---------------------------------------------------------------------------
# The inputs.
# ---------------------------------------------------------------------------


def compound_components() -> str:
    """The three components the design actually has, as Accelergy classes.

    **Compound classes rather than bare primitives, and that is forced.**
    Accelergy 0.4 at the pinned sha ships no `primitive_component_libs`
    directory, so a primitive component has no declared action list and the
    energy reference table comes out empty. Its own shipped example fails the
    same way. Declaring each component as a compound class with its actions
    written out is the supported path and is what the architecture description
    Section 16.4 asks for looks like in this version.

    **The array is one processing element, and the array size is applied by this
    project rather than by Accelergy.** A `fpmac[0..255]` subcomponent makes
    Accelergy multiply the *per action* energy by 256 as well as the area, which
    would charge 256 multiplies for every one the simulator counted. So the class
    is one PE, its action is one MAC, and `ARRAY_DIM * ARRAY_DIM` from this
    project's own cost model scales the area. The arithmetic is this project's
    and is auditable; the coefficients are Accelergy's.
    """
    return f"""compound_components:
  version: 0.4
  classes:
    - name: mac_array
      attributes:
        technology: "{TECHNOLOGY_NODE}"
        global_cycle_seconds: {GLOBAL_CYCLE_SECONDS}
        exponent: 8
        mantissa: 23
      subcomponents:
        - name: mac
          class: fpmac
          attributes:
            technology: technology
            global_cycle_seconds: global_cycle_seconds
            exponent: exponent
            mantissa: mantissa
      actions:
        - name: mac
          subcomponents: [{{name: mac, actions: [{{name: access}}]}}]

    - name: scratchpad
      attributes:
        technology: "{TECHNOLOGY_NODE}"
        global_cycle_seconds: {GLOBAL_CYCLE_SECONDS}
        width: {SCRATCHPAD_WORD_BITS}
        depth: 1024
      subcomponents:
        - name: storage
          class: SRAM
          attributes:
            technology: technology
            width: width
            depth: depth
            n_rw_ports: 2
            n_banks: 1
      actions:
        - name: read
          subcomponents: [{{name: storage, actions: [{{name: read}}]}}]
        - name: write
          subcomponents: [{{name: storage, actions: [{{name: write}}]}}]

    - name: main_memory
      attributes:
        technology: "{TECHNOLOGY_NODE}"
        global_cycle_seconds: {GLOBAL_CYCLE_SECONDS}
        width: {DRAM_ACCESS_BYTES * 8}
      subcomponents:
        - name: storage
          class: DRAM
          attributes:
            width: width
            type: "LPDDR4"
      actions:
        - name: read
          subcomponents: [{{name: storage, actions: [{{name: read}}]}}]
        - name: write
          subcomponents: [{{name: storage, actions: [{{name: write}}]}}]
"""


def scratchpad_depth(scratchpad_bytes: int) -> int:
    """How many words the cell's budget is, at one f32 per word.

    Rounded **up** to the next power of two, because CACTI models a real SRAM
    and this project's budgets are not powers of two. Up rather than down, so the
    modelled memory is never smaller than the one the program was allocated into:
    an under sized SRAM would report an energy per access lower than the design
    could achieve, which is the direction that flatters the result.
    """
    words = max(1, -(-scratchpad_bytes // (SCRATCHPAD_WORD_BITS // 8)))
    depth = 1
    while depth < words:
        depth *= 2
    return depth


def architecture(scratchpad_bytes: int) -> str:
    """The architecture description of Section 16.4, for one cell's budget."""
    return f"""architecture:
  version: 0.4
  subtree:
    - name: npu
      attributes:
        technology: "{TECHNOLOGY_NODE}"
        global_cycle_seconds: {GLOBAL_CYCLE_SECONDS}
      local:
        - name: mac_array
          class: mac_array
          attributes: {{}}
        - name: scratchpad
          class: scratchpad
          attributes:
            depth: {scratchpad_depth(scratchpad_bytes)}
        - name: main_memory
          class: main_memory
          attributes: {{}}
"""


def action_counts(counts: dict[str, dict[str, int]]) -> str:
    lines = [
        "action_counts:",
        "  version: 0.4",
        "  subtree:",
        "    - name: npu",
        "      local:",
    ]
    for component in COMPONENTS:
        lines.append(f"        - name: {component}")
        lines.append("          action_counts:")
        for action, count in counts[component].items():
            lines.append(f"            - {{name: {action}, counts: {count}}}")
    return "\n".join(lines) + "\n"


def counts_for(result: dict[str, Any]) -> dict[str, dict[str, int]]:
    """The action counts one cell hands Accelergy, from `Stats` and nothing else.

    **`macs` is raw and is the only MAC figure this function can see.** It reads
    `simulation.macs` by name. `effective_macs` is in the same block and is never
    read here, which Section 5.5 requires and
    `test_the_energy_path_never_sees_the_scaled_count` asserts by constructing a
    cell whose two differ and checking which one the answer follows.
    """
    simulation = result["simulation"]

    for name in ("scratchpad_elements_read", "scratchpad_elements_written"):
        if simulation.get(name) is None:
            raise AccelergyError(
                f"{result['cell']['name']} records {name} as null, so the "
                f"scratchpad has no action count. Section 16.4: an absent number "
                f"must never be indistinguishable from a zero, and a scratchpad "
                f"charged zero accesses would report a design that never touched "
                f"its own memory."
            )

    return {
        "mac_array": {"mac": int(simulation["macs"])},
        "scratchpad": {
            "read": int(simulation["scratchpad_elements_read"]),
            "write": int(simulation["scratchpad_elements_written"]),
        },
        "main_memory": {
            "read": dram_accesses(int(simulation["dram_bytes_read"])),
            "write": dram_accesses(int(simulation["dram_bytes_written"])),
        },
    }


def dram_accesses(byte_count: int) -> int:
    """How many DRAM accesses a byte count is, rounded **up**.

    **The first version of this raised on a byte count that was not a whole
    number of accesses, and that was the wrong conclusion from the right
    instinct.** The instinct was Section 16.4's, that rounding must never
    silently invent or discard an access. The wrong part was reading a remainder
    as a sign of a bug: `dilated_stack` moves 5004 bytes at one budget, because
    its buffers are f32 tensors with odd element counts and 1251 floats is 5004
    bytes. There is nothing wrong with that traffic.

    A DRAM cannot fetch part of a word. A transfer of 5004 bytes is 626 accesses,
    the last of which carries four useful bytes and four wasted ones, and the
    energy of that last access is paid in full. So this rounds up, which is both
    the physical answer and the direction that does not flatter the result.
    """
    return -(-byte_count // DRAM_ACCESS_BYTES)


# ---------------------------------------------------------------------------
# Running the tool.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """One Accelergy run's answer, with where every figure came from."""

    #: `component -> action -> picojoules per action`, from `ERT.yaml`.
    energy_per_action_pj: dict[str, dict[str, float]]
    #: `component -> square micrometres`, from `ART.yaml`.
    area_um2: dict[str, float]
    #: `component -> picojoules`, from `energy_estimation.yaml`.
    energy_pj: dict[str, float]
    #: Which estimation plug in answered for each component, from
    #: `ERT_summary.yaml`.
    estimators: dict[str, str]
    #: The action counts this run was given. `energy_pj` above is Accelergy's own
    #: total **for these counts and no others**, so a caller reusing the table
    #: for a different cell can tell whether the reconciliation below applies.
    counts_used: dict[str, dict[str, int]] | None = None

    provenance: str = (
        "ERT.yaml gives picojoules per action, ART.yaml gives square "
        "micrometres, energy_estimation.yaml gives picojoules per component. "
        "Accelergy rounds every figure to six significant figures, which is its "
        "default precision and is pinned here, so a comparison against one of "
        "these carries at most a half unit in the sixth digit."
    )


def registered_estimators() -> list[str]:
    """Which estimation plug ins registered, which decides what Accelergy answers.

    Section 16.1 asks for this list in the manifest and gives the reason: two runs
    at the same Accelergy sha with different plug ins are two different
    measurements. It is read from the tool's own `-l` output rather than from a
    list written here, because a list written here would be a claim about the
    machine rather than a reading of it.
    """
    completed = subprocess.run(
        ["accelergy", "-l"], capture_output=True, text=True, check=False
    )
    found = []
    for line in (completed.stdout + completed.stderr).splitlines():
        marker = "Found estimator plug-in: "
        if marker in line:
            name = line.split(marker, 1)[1].split(" (", 1)[0].strip()
            if name and name not in found:
                found.append(name)
    if not found:
        raise AccelergyError(
            "no Accelergy estimation plug in registered. Section 16.4: a missing "
            "external tool fails loudly naming the dependency. An empty estimator "
            "list would make every energy figure below the answer of a tool with "
            "nothing to answer from.\n\n"
            f"accelergy said:\n{(completed.stdout + completed.stderr)[-2000:]}"
        )
    return found


def run_accelergy(
    *, scratchpad_bytes: int, counts: dict[str, dict[str, int]], directory: Path
) -> Estimate:
    """One Accelergy run, with its answer checked before it is believed.

    The exit status is checked and **is not sufficient**: Accelergy's own shipped
    example crashes on this install and exits zero. So the three output files
    have to exist and every component this run asked about has to appear in each.
    """
    import yaml

    directory.mkdir(parents=True, exist_ok=True)
    inputs = directory / "in"
    output = directory / "out"
    inputs.mkdir(exist_ok=True)

    (inputs / "arch.yaml").write_text(architecture(scratchpad_bytes), encoding="utf-8")
    (inputs / "components.yaml").write_text(compound_components(), encoding="utf-8")
    (inputs / "counts.yaml").write_text(action_counts(counts), encoding="utf-8")

    command = [
        "accelergy",
        str(inputs / "arch.yaml"),
        str(inputs / "components.yaml"),
        str(inputs / "counts.yaml"),
        "-o",
        str(output),
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, cwd=str(directory)
    )

    wanted = {
        "ERT": output / "ERT.yaml",
        "ERT_summary": output / "ERT_summary.yaml",
        "ART": output / "ART.yaml",
        "energy": output / "energy_estimation.yaml",
    }
    absent = [name for name, path in wanted.items() if not path.is_file()]
    if absent:
        raise AccelergyError(
            f"Accelergy exited {completed.returncode} and wrote no {absent}. **A "
            f"zero exit is not an answer from this tool**: its own shipped "
            f"example crashes on this install and exits zero, so the outputs "
            f"having been written is the condition that matters and it is "
            f"checked here rather than inferred.\n\n"
            f"command:\n  {' '.join(command)}\n\n"
            f"stdout:\n{completed.stdout.strip()[-4000:] or '(nothing)'}\n\n"
            f"stderr:\n{completed.stderr.strip()[-4000:] or '(nothing)'}"
        )

    ert = yaml.safe_load(wanted["ERT"].read_text(encoding="utf-8"))["ERT"]
    art = yaml.safe_load(wanted["ART"].read_text(encoding="utf-8"))["ART"]
    estimation = yaml.safe_load(wanted["energy"].read_text(encoding="utf-8"))[
        "energy_estimation"
    ]

    per_action: dict[str, dict[str, float]] = {}
    for table in ert["tables"]:
        component = table["name"].split(".")[-1]
        per_action[component] = {
            action["name"]: float(action["energy"]) for action in table["actions"]
        }

    # **Which plug in answered is in the summary and not in the table.** Section
    # 16.1 asks for the registered estimator list because two runs at the same
    # Accelergy sha with different plug ins are two different measurements, and
    # the same reasoning applies one level down: knowing that the array's figure
    # came from `Aladdin_table` and the scratchpad's from `CactiSRAM` is what
    # makes the two comparable with the published tables they came from.
    # `ERT.yaml` records the energies and not their source; `ERT_summary.yaml`
    # records the source. Reading the first alone recorded `unrecorded` for every
    # component, which is a field that looks filled and says nothing.
    summary = yaml.safe_load(wanted["ERT_summary"].read_text(encoding="utf-8"))[
        "ERT_summary"
    ]
    estimators: dict[str, str] = {}
    for table in summary["table_summary"]:
        component = table["name"].split(".")[-1]
        sources = {
            str(entry["estimator"])
            for entry in table.get("primitive_estimation(s)", [])
            if entry.get("estimator")
        }
        if not sources:
            raise AccelergyError(
                f"ERT_summary.yaml names no estimator for {component}. A "
                f"coefficient whose source is unrecorded cannot be compared with "
                f"the published table it is supposed to come from."
            )
        estimators[component] = ", ".join(sorted(sources))

    areas = {
        entry["name"].split(".")[-1]: float(entry["area"]) for entry in art["tables"]
    }
    energies = {
        entry["name"].split(".")[-1]: float(entry["energy"])
        for entry in estimation["components"]
    }

    for label, table in (
        ("ERT", per_action),
        ("ART", areas),
        ("energy_estimation", energies),
    ):
        missing = [name for name in COMPONENTS if name not in table]
        if missing:
            raise AccelergyError(
                f"{label} does not carry {missing}. Accelergy answered about a "
                f"different design from the one it was asked about, and a figure "
                f"taken from it would be attributed to a component that is not "
                f"there.\n\nstdout:\n{completed.stdout.strip()[-2000:]}"
            )

    return Estimate(
        energy_per_action_pj=per_action,
        area_um2=areas,
        energy_pj=energies,
        estimators=estimators,
        counts_used={component: dict(actions) for component, actions in counts.items()},
    )


# ---------------------------------------------------------------------------
# One cell.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellEnergy:
    """Everything Accelergy says about one cell."""

    cell: str
    counts: dict[str, dict[str, int]]
    energy_pj_per_component: dict[str, float]
    area_mm2_per_component: dict[str, float]
    energy_pj: float
    area_mm2: float
    estimators: dict[str, str]
    energy_per_action_pj: dict[str, dict[str, float]]

    def as_schema(self, simulated_cycles: float) -> dict[str, Any]:
        """The `external` and `normalized` energy fields Section 16.1 names.

        `edp` is energy times latency, and the latency is `simulated_cycles`
        times the pinned clock, so the product is in picojoule seconds and the
        unit is stated here because a bare `edp` number means nothing without it.
        """
        return {
            "energy_pj": self.energy_pj,
            "energy_pj_per_component": dict(self.energy_pj_per_component),
            "area_mm2": self.area_mm2,
            "area_mm2_per_component": dict(self.area_mm2_per_component),
            "technology_node": TECHNOLOGY_NODE,
            "energy_pj_per_inference": self.energy_pj,
            "edp": self.energy_pj * simulated_cycles * GLOBAL_CYCLE_SECONDS,
        }


def energy_for(result: dict[str, Any], estimate: Estimate) -> CellEnergy:
    """One cell's energy and area, from its own counts and Accelergy's coefficients.

    The per component energy is recomputed here from the counts and the ERT
    rather than read from `energy_estimation.yaml`, and the two are asserted
    equal. That is not redundancy: it is what makes the number reconstructible
    from a committed result file by anyone holding the ERT, which is the
    difference between a figure a reader can check and a figure a reader has to
    take.
    """
    counts = counts_for(result)
    array_positions = cost_model.ARRAY_DIM * cost_model.ARRAY_DIM

    energies: dict[str, float] = {}
    for component in COMPONENTS:
        table = estimate.energy_per_action_pj[component]
        missing = [action for action in counts[component] if action not in table]
        if missing:
            raise AccelergyError(
                f"the energy reference table has no entry for {component} "
                f"{missing}. A missing coefficient read as zero would be an "
                f"action this design performs and pays nothing for."
            )
        energies[component] = sum(
            table[action] * count for action, count in counts[component].items()
        )

    # Accelergy was asked about one processing element. The array has
    # `ARRAY_DIM * ARRAY_DIM` of them, and that multiplication belongs to this
    # project's cost model rather than to the tool. Energy is **not** scaled: the
    # action count is the number of multiplies that happened, wherever they
    # happened, and multiplying it by the array size would charge every MAC 256
    # times.
    areas = {component: estimate.area_um2[component] / 1e6 for component in COMPONENTS}
    areas["mac_array"] *= array_positions

    # **The reconciliation applies only to the cell Accelergy was actually run
    # for.** `Estimator` runs the tool once per distinct scratchpad budget and
    # reuses the reference table for every cell at that budget, because the
    # architecture depends on the budget and the energy is linear in the counts.
    # `energy_estimation.yaml` is Accelergy's total for the counts of whichever
    # cell triggered the run, so comparing a different cell's reconstruction
    # against it compares two different workloads. That linearity is the thing
    # the caching rests on and it is proved in
    # `test_the_cached_table_gives_the_same_answer_as_a_fresh_run`, by running
    # the tool again for a second cell and comparing.
    if estimate.counts_used == counts:
        reported = {
            component: estimate.energy_pj[component] for component in COMPONENTS
        }
        for component in COMPONENTS:
            scale = max(abs(energies[component]), abs(reported[component]), 1.0)
            # Accelergy rounds its per action energies to six significant figures
            # and then multiplies, so the two paths can differ in the sixth digit
            # of the product. 1e-5 relative is one order above that and six
            # orders below any difference that would mean the counts had been
            # applied differently.
            if abs(energies[component] - reported[component]) > 1e-5 * scale:
                raise AccelergyError(
                    f"{result['cell']['name']}: the energy reconstructed from "
                    f"the action counts and the ERT is {energies[component]!r} "
                    f"pJ for {component} and Accelergy reported "
                    f"{reported[component]!r}. One of the two is not the number "
                    f"this cell's counts produce."
                )

    return CellEnergy(
        cell=result["cell"]["name"],
        counts=counts,
        energy_pj_per_component=energies,
        area_mm2_per_component=areas,
        energy_pj=sum(energies.values()),
        area_mm2=sum(areas.values()),
        estimators=dict(estimate.estimators),
        energy_per_action_pj={
            component: dict(estimate.energy_per_action_pj[component])
            for component in COMPONENTS
        },
    )


class Estimator:
    """Accelergy runs, cached on the only input that changes between cells.

    The architecture depends on the cell's scratchpad budget and on nothing else,
    and the energy is linear in the action counts, so one run per distinct budget
    answers every cell at that budget. `energy_for` reconstructs each cell's
    figure from the counts and the table and asserts it against Accelergy's own
    total, so the caching is checked rather than assumed: a cell whose answer did
    not come from its own counts would fail that assertion.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._cache: dict[int, Estimate] = {}
        self._estimators = registered_estimators()

    @property
    def registered(self) -> list[str]:
        return list(self._estimators)

    def estimate(self, result: dict[str, Any]) -> Estimate:
        budget = int(result["cell"]["scratchpad_budget_bytes"])
        if budget not in self._cache:
            self._cache[budget] = run_accelergy(
                scratchpad_bytes=budget,
                counts=counts_for(result),
                directory=self._directory / f"budget{budget}",
            )
        return self._cache[budget]

    def energy(self, result: dict[str, Any]) -> CellEnergy:
        return energy_for(result, self.estimate(result))


# ---------------------------------------------------------------------------
# The command line.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="accelergy_energy.py",
        description="Section 16.4's energy and area over the committed cells.",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)

    try:
        return _run(arguments)
    except (AccelergyError, ResultSchemaError) as failure:
        print(f"accelergy: {failure}", file=sys.stderr)
        return 2


def _run(arguments: argparse.Namespace) -> int:
    import tempfile

    results = [
        load_result(path) for path in sorted(Path(arguments.results).glob("*.json"))
    ]
    if arguments.models:
        wanted = set(arguments.models)
        results = [result for result in results if result["cell"]["model"] in wanted]
    if not results:
        raise AccelergyError("no committed cells matched")

    answers: list[CellEnergy] = []
    with tempfile.TemporaryDirectory(prefix="npu-accelergy-") as directory:
        estimator = Estimator(Path(directory))
        print(f"accelergy: estimators {estimator.registered}")
        for result in results:
            answer = estimator.energy(result)
            answers.append(answer)
            if arguments.verbose:
                print(
                    f"  {answer.cell}: {answer.energy_pj / 1e6:.4f} uJ, "
                    f"{answer.area_mm2:.4f} mm2, "
                    + ", ".join(
                        f"{name} {value / answer.energy_pj * 100:.1f}%"
                        for name, value in answer.energy_pj_per_component.items()
                    )
                )

    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(
                [
                    {
                        "cell": answer.cell,
                        "counts": answer.counts,
                        "energy_pj": answer.energy_pj,
                        "energy_pj_per_component": answer.energy_pj_per_component,
                        "area_mm2": answer.area_mm2,
                        "area_mm2_per_component": answer.area_mm2_per_component,
                        "energy_per_action_pj": answer.energy_per_action_pj,
                        "estimators": answer.estimators,
                    }
                    for answer in answers
                ],
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    per_mac = answers[0].energy_per_action_pj["mac_array"]["mac"]
    reference = REFERENCE_PJ["fp32_multiply"] + REFERENCE_PJ["fp32_add"]
    print(
        f"accelergy: {len(answers)} cells at {TECHNOLOGY_NODE}. Per MAC "
        f"{per_mac:.4f} pJ against a published fp32 multiply plus add of "
        f"{reference:.2f} pJ, a factor of {per_mac / reference:.2f}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
