# Controlled LoRA and Ranking Experiment

Status: **Phase 1 complete; protocol-locked confirmatory matrix evaluated**

Protocol date: October 18, 2026
Completion date: October 19, 2026

Implementation progress:

- Milestone 1 complete: protocol frozen and linked from the project documentation.
- Milestone 2 complete: core masked-residual, LoRA, MSE, and ListMLE components implemented;
  the repository test suite passes.
- Milestone 3 complete: validation-gated training, checkpointing, safe resume behavior, and both
  synthetic frozen-MSE and LoRA-ListMLE smoke paths are implemented and tested.
- Milestone 4 complete: all four validation-only five-epoch pilot runs completed on split 42;
  test positions remained untouched. ListMLE showed the clear validation gain, LoRA gradients were
  nonzero, and ranking throughput was identified as the blocker before multi-seed development.
- Milestone 5 complete: all 12 validation-only development runs completed on split 42. The
  protocol was locked before any confirmatory test evaluation (`SHA-256`
  `c7ad1f57448e136cbe19fae46d996370f520403eccfef591073640d6b3d4001b`).
- Milestone 6 complete: all 60 protocol-locked test evaluations completed across five
  position-held-out splits and three optimization seeds per configuration.
- Throughput decision: exact frozen-representation caching is implemented and verified. The
  expanded validation-only audit selected list microbatch size 4. It was fastest across the five
  pilot epochs on MPS and remained faster than size 8 in a direct CPU check. Early-stopping
  patience is 5 based on the completed pilot trajectories and feasibility analysis. Test
  positions remained untouched.

This document defines the next research phase for the TEM-1
`BLAT_ECOLX_Stiffler_2015` mutation-effect project. Its purpose is to isolate two
questions before introducing auxiliary assays or meta-learning:

1. Does a listwise ranking objective improve generalization relative to score regression?
2. Does constrained LoRA adaptation improve on an otherwise identical frozen ESM-2 model?

Changes made after pilot results must be recorded in the amendment log at the end of this
document. Changes made after confirmatory test evaluation require a new experiment rather than
an edit to the interpretation of this one.

## Scope and hypotheses

The experiment is an FSFP-inspired ablation, not a reproduction of the full FSFP method. It uses
target-assay training, LoRA, and a listwise ranking objective, but does not yet use MAML,
auxiliary DMS assays, or MSA-derived pseudo-labels.

- **H1 — objective effect:** ListMLE will improve held-out-position ranking relative to MSE when
  the backbone treatment is fixed.
- **H2 — adaptation effect:** LoRA will improve held-out-position ranking relative to a frozen
  backbone when the loss is fixed.
- **H3 — combined method:** LoRA plus ListMLE will outperform both zero-shot masked-marginal
  ESM-2 and the existing frozen ESM-2 ridge baseline on average across position splits.

All three hypotheses may be rejected. A negative result remains a valid outcome.

## Model definition

For a single mutation at reference position `i`, construct one input by replacing the wild-type
residue at `i` with ESM's mask token. ESM-2 produces the final-layer hidden representation
`h_i` at the masked position. A linear residual head maps `h_i` to 20 amino-acid residuals.

For wild-type amino acid `w` and alternative amino acid `a`, the prediction is:

```text
prediction(i, w -> a) = fixed_zero_shot(i, w -> a) + residual(i, a) - residual(i, w)
```

The fixed zero-shot term is the existing masked-marginal log-odds score:

```text
log p(a | position i masked) - log p(w | position i masked)
```

The residual head is initialized to zero. Before optimization, every configuration must therefore
reproduce the existing zero-shot predictions within numerical tolerance. One masked forward pass
can be shared by every observed substitution at the same position.

The base score and DMS target may be standardized using statistics computed from training rows
only. Validation and test labels must never contribute to those statistics.

## Controlled comparison

| Configuration | Backbone parameters | Training loss |
|---|---|---|
| Frozen + MSE | Frozen | Mean squared error |
| Frozen + ListMLE | Frozen | ListMLE, list size 5 |
| LoRA + MSE | LoRA adapters trainable | Mean squared error |
| LoRA + ListMLE | LoRA adapters trainable | ListMLE, list size 5 |

The masked-input construction, residual head, optimizer family, split, validation procedure, and
evaluation code must be identical except for the named experimental factor.

Required reference predictors are:

- Zero-shot masked-marginal ESM-2.
- Existing frozen ESM-2 wild-type-plus-difference ridge.
- Existing mutation-feature ridge.

## LoRA specification

- Backbone: `esm2_t6_8M_UR50D`.
- Apply adapters in all six transformer layers.
- Target modules: `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, and `fc2`.
- Rank: 16.
- Alpha: 16.
- Adapter dropout: 0.05.
- Freeze original ESM-2 parameters and its language-model head.
- Train the LoRA parameters and the residual prediction head only.

For the local fair-esm checkpoint, this adds 552,960 adapter parameters. The 320-to-20 residual
head adds 6,420 parameters, for 559,380 trainable parameters in the LoRA configuration. The
implementation must calculate and record these counts rather than relying only on these expected
values.

## Loss definitions

### MSE

Fit standardized training `DMS_score` values with mean squared error. Compute the target mean and
standard deviation from the training subset only. Since evaluation uses rank correlation,
predictions do not need to be transformed back to the original target scale for Spearman
calculation, but exported files should include clearly named raw model scores.

### ListMLE

Shuffle the training variants deterministically from the training seed and form lists of five.
Sort each list by its experimental `DMS_score` and minimize the negative Plackett-Luce
log-likelihood of that ordering. The current assay has unique `DMS_score` values, so the primary
experiment does not need a tie-breaking policy. The implementation must nevertheless fail with a
clear message or apply a documented deterministic rule if tied labels are later supplied.

Lists are sampled globally from training variants. Mixing global lists with within-position lists
would introduce another experimental factor and is outside this protocol.

## Split and data-use protocol

The primary result uses position-held-out splits only.

- Outer split seeds: `42 43 44 45 46`.
- Fractions: 80% training positions, 10% validation positions, 10% test positions.
- Every substitution at a residue position remains in the same subset.
- Neural models train on training positions only.
- Validation positions select the checkpoint and support predeclared development decisions.
- Test positions are evaluated once after the configuration is frozen.
- Neural models are not refitted on combined training and validation data.

The random-variant split remains a diagnostic and is not a headline LoRA result.

This repository's test results have already informed the broader project direction. Consequently,
this phase is a controlled development experiment on TEM-1 rather than a fully untouched external
confirmation. A separate verified assay must be reserved for later confirmation after model
selection is complete.

## Optimization protocol

Initial fixed settings:

| Setting | Value |
|---|---:|
| Optimizer | AdamW |
| Residual-head learning rate | `1e-3` |
| LoRA learning rate | `1e-4` |
| Weight decay | `0.01` |
| Maximum epochs | 100 |
| Early-stopping patience | 15 |
| Gradient-norm limit | 3.0 |
| List size | 5 |
| Initial lists per batch | 16 |
| Numeric precision | float32 |
| Device selection | automatic MPS, CUDA, or CPU |

Checkpoint selection uses validation overall Spearman correlation. If two epochs have equal
validation Spearman at stored precision, prefer the lower validation loss and then the earlier
epoch. Test metrics must not be printed or inspected during training and checkpoint selection.

Each final configuration uses three optimization seeds within each of the five position splits,
for 60 confirmatory training runs:

```text
4 configurations x 5 position splits x 3 optimization seeds = 60 runs
```

Optimization seeds measure training variability. They are not treated as 15 independent assay
replicates. The five position splits remain the primary experimental units.

The optimization seeds are fixed as `1 2 3`. The Milestone 5 development matrix uses split 42
only, giving 12 validation-only runs. The same three seeds are then reused unchanged across all
five splits in the confirmatory matrix.

## Evaluation and reporting

The primary metric is variant-level Spearman correlation on held-out test positions.

Secondary metrics are:

- Between-position Spearman correlation.
- Position-centered within-position Spearman correlation.
- Within-position pairwise accuracy.
- Macro within-position Spearman where defined.
- Validation Spearman and selected epoch.
- Runtime, device, and trainable parameter count.
- Within-split variation across optimization seeds.

For each configuration, report:

- Every individual run.
- Mean and sample standard deviation across optimization seeds within each position split.
- Mean, sample standard deviation, minimum, and maximum across the five split-level means.
- Paired per-split differences between controlled configurations and reference predictors.
- Number of position splits won.

Do not pool all 15 runs and present them as independent observations.

## Predeclared practical success criteria

LoRA plus ListMLE will be considered practically successful on this assay if it:

1. Exceeds zero-shot ESM-2 by at least 0.02 mean test Spearman across position splits.
2. Exceeds Frozen plus ListMLE under the same protocol.
3. Wins against zero-shot on at least four of five position splits.
4. Does not materially degrade the within-position ranking diagnostics.
5. Does not depend on one favorable optimization seed.

These are project decision criteria rather than a claim of statistical significance.

## Phase 1 results

All 60 runs in the locked confirmation matrix completed. For each split, the table first averages
the three optimization seeds; the reported standard deviation is then the sample standard
deviation across the five split-level means. The primary metric is held-out-position overall
Spearman correlation.

| Configuration | Mean | Split-level SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Frozen + MSE | 0.392251 | 0.140070 | 0.188505 | 0.545161 |
| Frozen + ListMLE | 0.435714 | 0.132173 | 0.292402 | 0.631315 |
| LoRA + MSE | 0.413071 | 0.156351 | 0.207186 | 0.622042 |
| LoRA + ListMLE | **0.473096** | **0.113617** | 0.372251 | 0.649853 |

LoRA + ListMLE is the strongest controlled configuration. Its mean advantage is `+0.037382`
over Frozen + ListMLE and `+0.035273` over the independently scored zero-shot ESM-2 reference
(`0.437823`). It beats Frozen + ListMLE on four of five splits; relative to zero-shot it wins
three splits, ties one, and loses one. Thus it satisfies the mean-improvement criteria but does
not satisfy the predeclared requirement to beat zero-shot on at least four splits. The proper
conclusion is evidence of a promising average improvement, not an unqualified success claim.

The ranking diagnostics are close to the frozen ListMLE comparator: LoRA + ListMLE has mean
within-position centered Spearman `0.402643` versus `0.412772` and pairwise accuracy `0.626976`
versus `0.630829`. Its strongest gain is between-position Spearman (`0.493835` versus `0.425322`).
Within-split optimization-seed variation was small for LoRA + ListMLE (split-level SD range
`0.000000`–`0.034826`), so its aggregate result does not arise from a single favorable
optimization seed.

The full, machine-readable run-level record is
`results/lora_confirmatory/confirmatory_summary.json`; the human-readable interpretation and
per-split table are in [`EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md).

## Milestones and stop gates

### Milestone 1 — Freeze the protocol

- Save this design document.
- Link it from the README and experiment log.
- Record that no LoRA results exist yet.
- Stop for review.

### Milestone 2 — Core implementation and unit tests

- Implement masked-position batching and residual scoring.
- Implement fair-esm-compatible LoRA linear layers.
- Implement MSE and ListMLE losses.
- Verify zero-initialized predictions, parameter freezing, gradient flow, and split isolation.
- Run the complete existing and new unit-test suite.
- Stop for code and test review.

### Milestone 3 — Training interface and dry run

- Add one configurable training entry point.
- Save configuration, checkpoints, histories, predictions, and metrics per run.
- Add resume-safe experiment directory handling.
- Run a tiny CPU/MPS smoke test only, not a scientific run.
- Stop before the pilot.

### Milestone 4 — Technical pilot

- Run all four configurations on split 42 with one optimization seed for five epochs.
- Inspect gradients, loss curves, validation behavior, runtime, memory, and numerical stability.
- Do not interpret pilot test performance as a scientific result.
- Record any protocol amendment required for feasibility.
- Stop before confirmatory training.

### Milestone 5 — Development and protocol lock

- Run the four configurations on split 42 with three optimization seeds.
- Use training and validation information only for permitted development decisions.
- Freeze the final configuration and record its checksum/configuration.
- **Complete.** All 12 validation-only runs completed, and the protocol lock was saved before
  confirmation.

### Milestone 6 — Confirmatory TEM-1 experiment

- Run the frozen 60-run matrix.
- Evaluate test positions exactly once per saved best checkpoint.
- Aggregate paired results and update `EXPERIMENT_LOG.md`.
- **Complete.** All 60 runs completed. LoRA + ListMLE had the highest five-split mean, but the
  strict four-of-five zero-shot win criterion was not met.

### Milestone 7 — Label-efficiency follow-up

- Run the winning method with 20, 40, 5%, 10%, 25%, 50%, and 100% of training labels.
- Use repeated nested subset samples.
- Compare with zero-shot and applicable frozen baselines.
- Identify the label regime in which adaptation becomes useful.

### Milestone 8 — Meta-learning extension

- Select and document related auxiliary ProteinGym assays.
- Establish no-meta baselines on the frozen 20- and 40-label TEM-1 subsets.
- Meta-train LoRA initialization on auxiliary tasks without target-test leakage.
- Adapt on the identical target subsets and isolate the contribution of meta-learning.

## Expected implementation artifacts

The exact module boundaries may change during implementation review, but the intended public
interfaces are:

```text
src/esm2_mutation_effects/adaptation.py
src/esm2_mutation_effects/lora.py
src/esm2_mutation_effects/ranking.py
src/esm2_mutation_effects/training.py
scripts/train_masked_adaptation.py
scripts/run_lora_development.py
scripts/lock_lora_protocol.py
scripts/run_lora_confirmatory.py
tests/test_adaptation.py
tests/test_lora.py
tests/test_ranking.py
tests/test_training.py
```

Generated checkpoints and metrics belong under `results/` and remain gitignored.

## Protocol amendment log

- **October 18, 2026 — computational batching decision:** Set the LoRA ListMLE list microbatch
  size to 8 for development and confirmation. Both candidates completed successfully with the
  same effective batch of 16 lists. Microbatch 8 was 43.6% faster per epoch than microbatch 16
  (`793.57` versus `1,407.77` seconds); their validation Spearman correlations differed by only
  `0.0000006`, and their training losses differed by `0.000078`. This changes only how the fixed
  effective batch is divided across forward/backward passes. It does not change the examples,
  loss, optimizer-step batch, split, checkpoint rule, or test policy. The selection used runtime
  and feasibility rather than validation quality. No test data were evaluated.
- **October 18, 2026 — superseding throughput correction:** The initial benchmark compared only
  sizes 8 and 16 and therefore omitted the pilot's original size 4. Auditing all five saved pilot
  epochs showed that size 4 required 279.89–357.79 seconds per LoRA+ListMLE epoch on MPS, versus
  793.57 seconds for size 8 and 1,407.77 seconds for size 16 in the later one-epoch benchmark. A
  direct CPU check also favored size 4 (`781.56` versus `805.70` seconds for size 8). Size 4 is
  therefore the final development and confirmation setting. The effective optimizer-step batch
  remains 16 lists, so this is a computational partitioning change rather than a change to the
  examples, loss, or number of optimizer updates. No test data were evaluated.
- **October 18, 2026 — early-stopping feasibility amendment:** Reduce patience from 15 to 5 while
  retaining the 100-epoch maximum. In the completed five-epoch pilot, the best validation epochs
  were 1 for both MSE arms and LoRA+ListMLE and 3 for Frozen+ListMLE; every subsequent observed
  validation Spearman value declined. Patience 5 preserves five full opportunities for recovery
  after the current best while making the predeclared 72-run remainder tractable. This decision
  used pilot validation trajectories and runtime only, before development or confirmatory test
  evaluation.
