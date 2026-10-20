# Phase 2: Label-Efficiency Experiment

Status: **corrected protocol ready for clean confirmation restart**

## Question

How many experimentally labelled TEM-1 single substitutions are needed before the Phase 1 winning
method, LoRA + ListMLE, improves meaningfully on zero-shot ESM-2 at completely unseen residue
positions? Does that improvement come from LoRA adaptation rather than the ranking objective alone?

This is a TEM-1 label-efficiency study, not an external generalization or meta-learning result.

## Fixed comparison

The two supervised methods use the Phase 1 architecture and optimization settings unchanged:

| Model | Backbone | Objective |
|---|---|---|
| Frozen + ListMLE | frozen ESM-2 t6 | ListMLE |
| LoRA + ListMLE | rank-16 LoRA ESM-2 t6 | ListMLE |
| Zero-shot ESM-2 | no fitted parameters | masked-marginal score |

The supervised settings are fixed from Phase 1: 100 maximum epochs, validation-Spearman checkpoint
selection, patience 5, list size 5, 16 lists per optimizer step, ListMLE microbatch size 4,
AdamW, head learning rate `1e-3`, LoRA learning rate `1e-4`, weight decay `0.01`, and gradient
norm limit 3. No Phase 2 test result may change these settings.

### October 20, 2026 numerical correction and restart

The first Phase 2 matrix was stopped after 146 of 190 runs when two distinct source DMS scores
became identical after float32 conversion in one ListMLE target list. The assay has no exact
float64 target ties; the failure was a numerical representation bug, not an experimental tie.
Those incomplete artifacts are retained separately and are not used for reporting.

The restarted protocol represents ListMLE targets as ordinal ranks derived from the original
float64 DMS scores. This is lossless for ListMLE, whose target values are used only to determine
their order, and the integer ranks are exactly representable as float32 on MPS. Regression target
scaling, model precision, subset manifests, data partitions, and every optimization setting remain
unchanged. A new lock is required before any restarted test evaluation.

### Computational optimization before restart

Frozen-backbone jobs use the same ESM-2 checkpoint, target sequence, masked residue positions, and
representation layer in every split and label-budget run. The runner therefore persists that
fixed site-representation tensor once per compute-device type under `data/processed/` and reuses
it across frozen jobs. The cache records checksums of its token inputs and representations plus
the model name, layer, tensor shape, dtype, and source device. A mismatch or corrupted checksum
fails instead of silently reusing the tensor. This changes only repeated computation: training
examples, predictions, losses, checkpoint selection, and test policy are unchanged. LoRA runs
cannot use this cache because their backbone representations update during training.

## Data partitioning and label budgets

- Outer evaluation: the five existing deterministic position-held-out splits, seeds 42–46.
- Every substitution at a residue remains in its original train, validation, or test partition.
- Training labels are individual mutation rows drawn only from the training positions.
- Within each outer split, each sampling seed generates one permutation; smaller budgets are
  prefixes of that permutation and are therefore nested.
- Sampling seeds: 1, 2, and 3, combined deterministically with the outer split seed.
- Fixed budgets: 20, 40, 200 (approximately 5%), 401 (approximately 10%), 1,002 (25%),
  2,004 (50%), and 4,008 (100%) labelled training variants.

The 100% budget is identical for all sampling seeds and will be evaluated once per outer split.
Budgets below 100% have 15 runs per supervised configuration (five outer splits times three nested
subset samples). This yields 95 runs per supervised configuration and 190 supervised runs total.
Zero-shot is evaluated once per outer split and reused as the no-label reference.

The Phase 1 training seed is fixed to `1` for new partial-label runs. This is deliberate: Phase 1
showed small within-split optimization-seed variation for LoRA + ListMLE (0.000000–0.034826), while
Phase 2's repeated sampling seeds estimate the uncertainty introduced by the scarce labelled set.
The result will explicitly distinguish subset-sampling variation from outer-split variation. The
three Phase 1 optimization seeds remain the full-label reference, rather than being treated as
additional independent data splits.

## Leakage safeguards

For every partial-label run:

1. The subset manifest contains only mutation IDs from the outer split's training positions.
2. The target standardizer is fitted only on the chosen subset—not on unused training labels,
   validation labels, or test labels.
3. The model's training loss uses only the chosen subset.
4. The full validation position set selects the checkpoint; validation labels are not added to the
   training subset and are reported separately from the label budget.
5. The held-out test positions are evaluated exactly once after the validation-selected checkpoint
   is saved.
6. Each saved subset manifest is fingerprinted and incorporated into the run configuration, so a
   mismatched resume cannot silently reuse a checkpoint.

Validation labels are a fixed selection overhead, not part of the displayed training-label budget.
This is the same accounting convention as the existing ridge label-efficiency experiment.

## Reporting

The primary outcome at each budget is test overall Spearman correlation. Report:

- each run and its subset manifest fingerprint;
- mean, sample standard deviation, minimum, and maximum across split-level means;
- the mean within-split standard deviation across nested subset samples;
- between-position Spearman, within-position centered Spearman, pairwise accuracy, and macro
  within-position Spearman;
- paired differences between LoRA + ListMLE, Frozen + ListMLE, and zero-shot at each outer split;
- the smallest budget at which LoRA + ListMLE has a pre-specified practical mean advantage of 0.02
  over zero-shot, while clearly stating its split-level uncertainty.

Do not pool repeated subsets as if they were independent biological assays. The five outer position
splits are the primary evaluation units.

## Milestones

1. **Subset-aware infrastructure:** persist and validate nested mutation-ID manifests; ensure the
   restricted subset controls target scaling and training loss; add tests.
2. **Validation-only smoke check:** use split 42 and a 20-label subset without test evaluation to
   verify count, manifest, standardizer, gradient flow, artifacts, and resume behavior.
3. **Protocol lock:** record the exact budgets, seeds, comparators, command version, and checksum
   before Phase 2 test evaluation.
4. **Confirmatory matrix:** execute the fixed matrix, aggregate results, and update the experiment
   log.

Meta-learning remains out of scope until this experiment establishes whether target-label scarcity
is the limiting factor for the Phase 1 method.
