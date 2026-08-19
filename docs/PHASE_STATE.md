<!--
SPDX-FileCopyrightText: 2026 Olajide Badejo <olajideayomidebadejo@gmail.com>

SPDX-License-Identifier: MIT
-->

# Phase state

*Diataxis type: reference.*

Ground rule 17: this file is updated at the end of **every** session, including
a session that achieved nothing, and it carries four things: the current phase,
the status of its gate, the open questions, and the exact next command. This
build spans dozens of sessions, and reconstructing where it stood from `git log`
costs more than writing these lines did.

**Last updated:** 2026-08-19.

## Current phase

**P3, the ONNX frontend and the model suite.** Branch
`phase/p3-onnx-frontend`, cut from `main` at `316f3b8`, which is the P2 merge.
Nine commits, not pushed. The ninth is the one that carries this table, so it
is the branch tip and is named by subject rather than by a sha it cannot know.

| Commit | Subject |
|---|---|
| `4b289bd` | `build(deps): install onnxscript and relock the environment` |
| `0a8bd84` | `docs(adr): record how the frontend emits npu dialect IR from Python` |
| `0d23a76` | `feat(dialect): let add and mul take the rank 1 channel broadcast operand` |
| `bf1dc82` | `feat(frontend): add the ONNX importer for the opset 23 operator set` |
| `677aeb6` | `feat(frontend): add the seeded model suite of Section 15` |
| `98603d5` | `ci: switch on mypy and pytest` |
| `fa71f43` | `docs: record the P3 defects and hand off the phase` |
| `16724ea` | `docs(readme): correct a phase line that has been wrong since P1` |
| tip | `docs: refresh the phase state commit table` |

A note on the branch point, since the previous phase had a scare about exactly
this. Local `main` was at the P1 merge `6da2a3f` when this session began; a
fetch showed `origin/main` at `316f3b8`, the P2 merge, and local `main` was fast
forwarded to it before the branch was cut. P2's lesson held: the conclusion was
drawn against `origin/main` after a fetch, not against a local ref.

## Gate status

The P3 gate is Section 23's, and every item is **met locally**. Item by item,
with the test that proves it. Every test named runs in
`python -m pytest test/Python -q`.

| Gate item | Proof |
|---|---|
| A pytest per converter in isolation | `test_onnx_importer.py` carries at least one positive case per registered converter: `test_conv_imports_with_its_bias_as_an_operand`, `test_gemm_with_transb_transposes_the_constant_at_import`, `test_matmul_is_rank_two_by_rank_two`, `test_same_shaped_operands_pass_straight_through`, `test_the_carve_out_reaches_mul_as_well_as_add`, `test_relu_imports`, `test_clip_with_a_zero_lower_bound_and_no_upper_bound_is_a_relu`, `test_a_chain_of_identities_is_imported_past_rather_than_refused`, `test_max_pool_imports_with_its_window`, `test_average_pool_with_count_include_pad_and_no_pads_is_accepted`, `test_global_average_pool_becomes_a_full_extent_pool`, `test_batch_normalization_imports_with_its_epsilon`, `test_a_flattening_reshape_keeps_the_batch`, `test_flatten_at_axis_one_keeps_the_batch`, `test_transpose_imports_with_its_permutation`, `test_concat_normalises_a_negative_axis` |
| Plus the negative cases | Twenty six refusal tests, covering `auto_pad`, a `kernel_shape` disagreeing with the filter, a non constant filter, an impossible convolution extent, `alpha`, `transA`, a batched `MatMul`, a run time broadcast, relu6 and a non zero `Clip` lower bound, `MaxPool` indices, `count_include_pad` with real pads, `AveragePool` dilations, the batch norm training form and non constant parameters, a batch folding `Reshape`, `Flatten` at axis 0, `allowzero` with a literal zero, a non permutation `perm`, a dynamic extent, an integer input, another opset, a custom operator domain, the quantization pair, `Pad`, and an operator with no converter at all |
| A structural pytest per model | `test_the_exported_graph_has_exactly_the_expected_nodes`, parametrized over all seven models, asserting **exact** node counts rather than a subset, because a subset check passes on a graph that gained a node |
| Exactly one `Mul` surviving export in the ResNet block | `test_exactly_one_mul_survives_the_resnet_export`, plus `test_the_resnet_scale_is_a_channel_shaped_initializer` |
| One `Transpose` in the dilated stack | `test_exactly_one_transpose_closes_the_dilated_stack`, which also asserts `perm = [0, 2, 3, 1]` |
| The determinism test | `test_two_exports_at_the_same_seed_produce_identical_first_layer_weights` over all seven models, bit exact on the first rank 4 initializer, plus `test_a_different_seed_produces_different_weights` so a generator that ignored its seed cannot pass, plus `test_the_hand_built_models_are_deterministic_too` byte comparing the serialized model |
| The batch preserving flatten test | `test_a_flattening_reshape_keeps_the_batch` and `test_a_reshape_that_folds_the_batch_away_is_refused`; `test_flatten_at_axis_one_keeps_the_batch` and `test_flatten_at_axis_zero_is_refused_by_name`; and `test_every_model_keeps_its_batch_dimension_end_to_end` over the whole suite |
| The docstring agreement test | `test_the_module_docstring_lists_exactly_the_registered_converters`, asserted in both directions |
| `Identity, Identity, Conv, BatchNormalization` imports past every `Identity` | `test_a_chain_of_identities_is_imported_past_rather_than_refused`, which asserts the convolution reads `%arg0` directly. The same sequence is at the head of `conv_bn_relu_stack`, so it is covered by a suite model as well as by a fixture |
| A `Conv` followed by a rank 1 `Add` imports to a rank 1 addend | `test_a_conv_followed_by_a_rank_one_add_imports_to_a_rank_one_addend`, which also asserts the convolution has **no** bias operand, so `-npu-fuse-bias` has something left to fuse rather than a convolution the importer already completed |
| Every compute operation carries a `tensor.empty` destination | `test_every_compute_operation_has_a_tensor_empty_destination` on a fixture and `test_every_compute_operation_in_every_model_has_its_own_destination` over all seven models. Both compare the sorted list of `outs(...)` operands against the sorted list of `tensor.empty` results, so a missing destination and a reused one both fail |

### Verification output

Every command below was run on this branch at `98603d5`, from
`/home/elijah/npu-mlir-v2`, in `~/npu-venv`, with `PYTHONPATH` unset in the
calling environment.

| Command | Result |
|---|---|
| `ninja -C build -j6` | clean, no warnings |
| `ninja -C build check-npu` | 7 discovered, 7 passed, 0 failed |
| `python -m pytest test/Python -q` | 142 passed, 7 deselected, exit 0 |
| `python -m pytest test/Python -q -m "not slow"` | 142 passed, 7 deselected |
| `python -m pytest test/Python -q -m "slow or not slow"` | 149 passed |
| `mypy` | no issues found in 11 source files |
| `black --check .` | 19 files unchanged |
| `ruff check .` | all checks passed |
| `bash scripts/dash-lint.sh` | `dash-lint: clean` |
| `bash scripts/dash-lint.sh --self-test` | 8 of 8 expectations met |
| `reuse lint` | compliant, 116 of 116 files |
| `pre-commit run --all-files` | all twelve hooks passed |
| `python scripts/check-reachability.py --skip-models` | pass, exit 0, and the **import layer is decidable for the first time**: 12 imported computation operations, every one found in `op_mapping.py` |
| `python scripts/gen-design-decisions.py --check` | index up to date |
| `.github/workflows/ci.yml` | parses as YAML; job list and step order confirmed |
| `git status --short` | empty |

Two extra verifications that are not gate items but are the evidence behind
decisions taken this phase.

- **`bf1dc82` stands alone.** Checked out into a detached worktree and run on
  its own: 76 passed, mypy clean. The importer commit does not depend on the
  model generator commit that follows it, which is what makes the two a
  sequence rather than one change split for appearances.
- **The lock file installs on the CI image's interpreter.** A `--dry-run`
  install of `requirements-lock.txt` inside `ubuntu:24.04`, which is what the
  LLVM image is built from, resolves every pin on Python 3.12.3, including
  `torch-2.13.0+cpu-cp312`. That is the single largest risk in the new pytest
  step and it is measured rather than assumed.

## Activation proofs, and the exact perturbations

Two steps switched on this phase, and Section 19.0 requires each to be broken
once deliberately, shown red, and restored. Both recipes below break an
**assertion** rather than the build, so each turns exactly one step red and
leaves the rest of the pipeline green, which is what makes the proof say
something about that step rather than about the compiler.

**mypy.** In `python/npu_frontend/onnx_importer.py`:

```
-PINNED_OPSET: Final[int] = 23
+PINNED_OPSET: Final[str] = 23
```

The value is unchanged, so pytest, black and ruff all stay green and the C++
build is untouched. Only mypy speaks. Verified locally: it reports two errors,
the incompatible assignment itself and a knock on `arg-type` where
`model_generator.py` passes the constant to `make_opsetid`, which is mypy
following the type across the package and is a better proof of the step than
one error would have been. pytest reported 142 passed on the same tree.

**pytest.** In `test/Python/test_model_generator.py`:

```
-    assert node_counts(suite["resnet_block"]).get("Mul") == 1
+    assert node_counts(suite["resnet_block"]).get("Mul") == 2
```

One structural assertion, on the model that carries the only `Mul` in the suite.
Verified locally: exactly one test fails, `mypy` stays green, and nothing else
moves.

**The exit 5 arm, which is a separate claim and deserves its own perturbation.**
The activation table says an empty collection returns 5 and is never read as a
pass, and a green pytest step does not prove that arm works. In
`pyproject.toml`:

```
-testpaths = ["test/Python"]
+testpaths = ["docs"]
```

pytest then collects nothing, exits 5, and the step must fail with the
`::error::pytest collected no tests and exited 5.` line rather than passing.
Locally that arm is already proved: `python -m pytest docs -q` exits 5.

## Open questions

Five. None blocks the gate.

**The `npu.add` and `npu.mul` relaxation is a P1 dialect change made at P3, and
a reviewer should look at it as one.** The reasoning is in
`docs/adr/0005-channel-broadcast-on-add-and-mul.md` and the conflict it resolves
is D-0012. The short version: P1's rule made Section 11's carve out
unimplementable for `Add`, because folding the addend into the convolution at
import leaves `-npu-fuse-bias` nothing to fuse, and unrepresentable for `Mul`,
because a per channel scale has no bias operand to be folded into. A frontend
cannot be written against a dialect that refuses the IR the specification asks
it to produce. It obliges P4 and P7: an add or a multiply with a rank 1 rhs
lowers to a per channel broadcast, and the kernel reads that operand with a
channel stride of one and a spatial stride of zero. Both are already implied by
`npu.batch_norm`, which is why the relaxation is cheaper than it looks, but
neither should be discovered from a verifier failure.

**`GlobalAveragePool` does not come from the model Section 15 assigns it to.**
The dynamo exporter on torch 2.13 lowers every spelling of adaptive average
pooling to `ReduceMean`, which is not in this project's operator set, so the
depthwise separable block pools with an `AveragePool` whose kernel is the full
spatial extent. That is global average pooling and it imports to the same
`npu.avg_pool2d`, so nothing the pipeline sees is different, but the ONNX node
type differs from what Section 15 implies and the `GlobalAveragePool` converter
gets its suite model in the conv plus batch norm stack instead. Recorded rather
than quietly done, because a reader comparing the suite against Section 15's
table will notice the difference and should find the reason here.

**mypy checks `python/npu_frontend` and `scripts`, not `test/Python`.** That is
P0's configuration and the activation table guards the step on
`python/npu_frontend/` existing, so the scope and the guard agree and nothing
was weakened here. Widening it to the tests would be strictly better and is real
work: the fixtures would need annotating under `disallow_untyped_defs`, and the
lint job would need `pytest` installed for its `py.typed` marker. Left as a
decision for whoever next touches the mypy configuration rather than taken
silently in a phase that had no reason to take it.

**The five input classes and the tolerance work of Section 17.4 are not here,
and should not be.** This phase produces models and IR; there is no simulator to
compare against until P7 and no end to end pipeline until P8. The `slow` marker
carries seven cells today, `test_every_model_imports_at_a_second_seed`
parametrized over the suite, so the fast and slow split is real rather than
declared and the CI step that runs the slow cells at P10 will have something to
run. The matrix that marker is really for arrives with it.

**`experiments/models/` does not exist, and `check-reachability.py` reads it for
the model layer.** Models are generated into a caller supplied directory and
`.onnx` files are never committed, so there is no directory for the full check to
point at. The full check is off until P8 per the activation table and the
`--skip-models` form is what runs today, so nothing is broken. But whoever turns
the full check on at P8 has to decide where the model layer looks: either the
benchmark harness generates into `experiments/models/` before the check runs, or
`LAYER_HOMES["model"]` moves to the generator's registry in
`python/npu_frontend/model_generator.py`, which is where the answer actually
lives.

## Next command

```bash
git push -u origin phase/p3-onnx-frontend
```

Then watch CI. Expect `lint` to run `mypy` for the first time, and
`build-and-test` to install the Python dependencies and run `pytest` for the
first time; that install is a few hundred megabytes on the first run and cached
on every one after, keyed on the lock file's hash. Then perform the two
activation proofs above, each as its own pull request so the `pull_request`
trigger fires, record which job and which step caught each in
`docs/ENGINEERING_LOG.md` with the run URLs, revert them, and open the merge
pull request.

**No image republish is needed.** P2 switched the MLIR Python bindings on in the
image and they are at `/opt/llvm/python_packages/mlir_core`, already on
`PYTHONPATH` there. The new pytest step installs its Python dependencies into
the running container from `requirements-lock.txt` rather than needing them
baked into the image.

## Next phase

**P4, lowering to `npuisa`.** `ConversionTarget`, `TypeConverter`,
`applyPartialConversion`, with the `one-shot-bufferize` attempt made and its
outcome recorded either way. Two things P3 leaves on P4's desk beyond the
roadmap entry: `npu.add` and `npu.mul` with a rank 1 rhs need a per channel
broadcast lowering, and `test/Dialect/NPUISA/dma-boundaries.mlir`, which Section
8 names and which P2's handoff correctly deferred to the phase that has a
lowering to assert against.

## The frozen v1 fallback

Recorded here because the P0 gate requires it in this file and in the P0 decision
record both, and repeated at every phase because a fact that stops being
repeated is a fact somebody eventually does not know.

- **Path:** `/home/elijah/npu-mlir`
- **HEAD:** `99408bc14b4f6331ce03ebf1dc0aecce1529afa8`
- **Dirty state:** only the untracked `upgrade_parts/` directory, which stays
  behind deliberately and is not needed by this build.

**Nothing in this project may ever write to that directory.** No phase, no
script, no tool, no agent, not once. It may be read, and only through a command
that cannot write. **Only the owner may retire it.**

The reason it exists on top of git history is that the two protect against
different failures. History protects against a bad commit. A second directory
protects against everything else, because if this rebuild goes wrong at any
point, deleting `~/npu-mlir-v2` returns the machine exactly to its pre build
state with no reasoning about reflogs required. That guarantee holds only while
the frozen copy is untouched.
