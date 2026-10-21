# Phase 2: Label-Efficiency Experiment

Status: **complete — 190/190 supervised runs validated**

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
selection, patience 5, list size 5, 16 lists per optimizer step, AdamW, head learning rate `1e-3`,
LoRA learning rate `1e-4`, weight decay `0.01`, and gradient norm limit 3. No Phase 2 test result
may change these settings. The ListMLE microbatch is a device-specific systems parameter selected
from 4, 8, and 16 by a one-epoch validation-only benchmark on the destination hardware before the
protocol lock. It does not change the effective batch of 16 lists, and the selected value is
recorded in the lock.

A direct validation-only MPS benchmark after the numerical correction confirmed the local
microbatch setting. On the same 2,004-label subset, sizes 4, 8, and 16 required 129.57, 163.63,
and 4,104.25 seconds for one epoch, respectively, while validation Spearman differed by at most
0.000031. No test positions were evaluated. CUDA batching will be treated as a device-specific
throughput choice and benchmarked separately before any Colab confirmation run.

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

Best and latest checkpoints store only parameters that are optimized: the residual head and, for
LoRA runs, the adapters. The named public ESM-2 backbone is reconstructed at load time. This avoids
repeating the frozen 8M-parameter backbone in every matrix checkpoint; the loader remains backward
compatible with earlier full-state checkpoints.

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

## Results

The completed CUDA protocol has SHA-256
`ecd213c9645890ca90656059867916c78e3a879cec72e9ebf1b46ca2d60cddad`. It used the
validation-only L4 benchmark's selected ListMLE microbatch of 16. All 190 expected supervised run
identities were present exactly once, with 95 Frozen + ListMLE and 95 LoRA + ListMLE runs. Every
primary test metric was finite. The analyzed `phase2_summary.json` has SHA-256
`35d817b5095db84e4bfb301428dac18997559439f9e2308f48cf4c84707c5d92`.

The table reports the mean and sample standard deviation across the five held-out-position split
means. Each split mean averages its three subset samples below the full-label budget.

| Labels | Frozen + ListMLE | LoRA + ListMLE | LoRA − zero-shot |
|---:|---:|---:|---:|
| 20 | 0.426817 ± 0.128307 | 0.425522 ± 0.128105 | -0.012300 |
| 40 | 0.427675 ± 0.132888 | 0.426235 ± 0.134590 | -0.011586 |
| 200 | 0.441302 ± 0.131955 | 0.442547 ± 0.133761 | +0.004726 |
| 401 | 0.436298 ± 0.124218 | 0.442377 ± 0.127471 | +0.004555 |
| 1,002 | 0.440744 ± 0.133436 | 0.450552 ± 0.127444 | +0.012730 |
| 2,004 | 0.440929 ± 0.123726 | **0.466493 ± 0.117322** | **+0.028671** |
| 4,008 | 0.436906 ± 0.130708 | 0.461451 ± 0.119521 | +0.023630 |

Zero-shot ESM-2 has mean test Spearman `0.437822 ± 0.141027` across the same five splits. The
pre-specified practical mean advantage of `+0.02` is first reached at 2,004 labels. At that budget,
LoRA also exceeds Frozen + ListMLE by `+0.025564`. The paired LoRA-minus-zero-shot split outcomes
are three wins, one tie, and one loss, with differences ranging from `-0.007343` to `+0.076464`.
The result is therefore a qualified mean improvement, not uniform superiority across held-out
position sets.

The representation benefit is mainly between positions. At 2,004 labels, LoRA's mean
between-position Spearman is `0.482758` versus frozen's `0.456228`, while its position-centered
Spearman is slightly lower (`0.389073` versus `0.399127`). This matches Phase 1's conclusion that
LoRA primarily helps identify which unseen positions are sensitive rather than improving every
substitution ordering within a position.

Position-split variation remains the main source of uncertainty. For LoRA at 2,004 labels, the
sample standard deviation across split means is `0.117322`, compared with mean within-split subset
sampling standard deviation `0.014464`. Which residue positions are held out matters far more than
which labeled mutations are selected within the training positions.

## Milestones

1. **Subset-aware infrastructure:** persist and validate nested mutation-ID manifests; ensure the
   restricted subset controls target scaling and training loss; add tests.
2. **Validation-only smoke check:** use split 42 and a 20-label subset without test evaluation to
   verify count, manifest, standardizer, gradient flow, artifacts, and resume behavior.
3. **Protocol lock:** record the exact budgets, seeds, comparators, command version, and checksum
   before Phase 2 test evaluation.
4. **Confirmatory matrix:** complete; all 190 supervised runs were aggregated and validated.

Phase 2 shows that target-label scarcity is limiting: 20–40 labels underperform zero-shot, while
the LoRA advantage emerges at 1,002 labels and crosses the practical threshold at 2,004. The next
research phase can now introduce auxiliary assays and test whether meta-learning or multi-assay
adaptation shifts this curve left.
