# Phase 3B (deferred): cross-assay meta-learning for label-efficient mutation prediction

**Status:** deferred complete research plan. The deadline-bounded active work is
[Phase 3A: one-day pilot](PHASE3A_ONE_DAY_PILOT.md). This document preserves
the original complete design for later execution; no Phase 3B assay data have
been downloaded, no Phase 3B code has been implemented, and no Phase 3B model
has been trained.

**Original design date:** October 20, 2026  
**Revision:** incorporates strict total-label accounting, an exact Reptile objective, adaptation-
based meta-validation, biologically related auxiliary controls, and fresh-protein confirmation.

## 1. Motivation from Phases 1 and 2

The existing target is the ProteinGym assay `BLAT_ECOLX_Stiffler_2015` (TEM-1 beta-lactamase).
Phase 1 established that LoRA combined with ListMLE was the strongest controlled target-only
configuration on average. Phase 2 then showed that its advantage was label-dependent:

- at 20 and 40 TEM-1 training labels, target-only LoRA underperformed zero-shot ESM-2;
- at 200--1,002 training labels, the mean advantage was small;
- the pre-specified `+0.02` mean advantage over zero-shot first appeared at 2,004 training labels;
- most of the gain came from ranking the sensitivity of different positions, not from consistently
  improving substitution ordering within a position.

Phase 3 asks whether experience from other DMS assays can produce an initialization that adapts to
a new protein with substantially fewer target-protein labels.

## 2. Research questions and hypotheses

### TEM-1 case-study question

Can cross-assay meta-learning shift the TEM-1 label-efficiency curve left while preserving strict
generalization to unseen residue positions?

### Primary TEM-1 hypothesis

At a **strict total budget of 401 accessible TEM-1 labels**, a Reptile-initialized LoRA model will
improve mean held-out-position Spearman correlation by at least `0.02` over standard-initialization
LoRA adapted with the identical labels and schedule, with a positive paired difference on at least
four of five new outer splits.

The 401-label budget is primary because it is approximately 10% of the former Phase 2 training pool
and is well below the 2,004-training-label threshold observed in Phase 2.

### Cross-protein confirmation question

Does the same locked method improve few-shot adaptation on protein tasks that were never used for
architecture choice, meta-training, meta-validation, or target-specific method selection?

### Secondary hypotheses

1. The smallest strict target-label budget at which meta-learned LoRA exceeds zero-shot ESM-2 by
   `0.02` will be lower than the corresponding standard-initialization threshold.
2. Meta-learning will outperform ordinary pooled multi-assay pretraining at matched auxiliary data
   and approximate compute, isolating the meta-objective from the benefit of more labeled assays.
3. Any overall gain will not be accompanied by a mean decrease greater than `0.02` in
   position-centered Spearman relative to standard-initialization LoRA.
4. A head-only meta-learning control will reveal whether few-shot transfer requires adapting ESM-2
   representations or mainly a better lightweight prediction head.

These are effect-size criteria, not claims of statistical significance. Paired protein- and split-
level results will be shown directly rather than relying on asymptotic tests with very small sample
counts.

## 3. Strict target-label accounting

This revision distinguishes **training-label budgets** from **total accessible target-label
budgets**.

A target label counts against the budget if its `DMS_score` is accessed for any of the following:

- gradient-based adaptation or fitting;
- target-specific early stopping or checkpoint selection;
- target-specific hyperparameter or learning-rate selection;
- target-specific model comparison or calibration before final evaluation.

Held-out test measurements do not count as labels available to the learner: they are opened only
after protocol lock to evaluate the retrospective benchmark. In a real deployment those test
measurements would not be required to train or rank new candidates.

### Experiment A -- strict total-label budget (primary)

- The declared budget is the complete set of target labels visible before final evaluation.
- There is **no additional labeled target-validation set**.
- Target adaptation uses a fixed number of steps and a fixed checkpoint rule selected on auxiliary
  meta-validation proteins before any target evaluation.
- All budgeted target labels may therefore be used for adaptation.
- A sensitivity analysis may perform cross-validation entirely inside the same budget, but it may
  not access additional target labels.
- All methods receive identical target mutation identifiers at a given budget.

Thus, “401 labels” means exactly 401 accessible target measurements, not 401 training measurements
plus several hundred uncounted validation measurements.

### Experiment B -- fixed-validation diagnostic (secondary)

The former Phase 2 convention may be retained as a secondary diagnostic:

- the displayed number is explicitly called **training labels**;
- the size of the additional labeled validation set is displayed beside it;
- results cannot be used to support a strict wet-lab label-efficiency claim.

Experiment B exists only to connect Phase 3 to the earlier learning curves and to measure the effect
of target-specific early stopping. Experiment A is the primary few-shot result.

## 4. Claim boundary and two-stage evaluation

New random partitions of TEM-1 are not an independent biological test. TEM-1 measurements have
already influenced the choice of masked-marginal residual scoring, ListMLE, LoRA, and the decision
to pursue meta-learning.

Phase 3 therefore has two explicitly different stages:

1. **Phase 3A -- TEM-1 extended evaluation.** New position partitions test whether auxiliary
   initialization helps this established target under a stricter label budget. This is a controlled
   target-protein case study, not independent cross-protein confirmation.
2. **Phase 3B -- fresh-protein confirmation.** Five protein groups are reserved before training.
   They are untouched by meta-training, meta-validation, architecture choice, and target-specific
   method selection. They provide the evidence needed for a cross-protein transfer claim.

A positive TEM-1 result alone supports only a TEM-1-specific conclusion. A generalizable
meta-learning claim requires a positive result on the reserved protein groups under the locked
protocol.

## 5. Controlled model comparison

All methods use `facebook/esm2_t6_8M_UR50D`, masked-marginal mutation scores, higher-is-better
ProteinGym targets, and evaluation on residue positions absent from target adaptation.

| Method | Auxiliary labels | Target labels | Role |
|---|---:|---:|---|
| Zero-shot ESM-2 | No | 0 | Fixed no-label reference |
| Standard initialization -> LoRA + ListMLE | No pretraining | Same strict budget | Target-only adaptation control |
| Pooled multi-assay pretraining -> LoRA + ListMLE | Yes | Same strict budget | Controls for additional supervised data |
| Reptile initialization -> LoRA + ListMLE | Yes | Same strict budget | Tests explicit fast-adaptation training |
| Head-only Reptile -> frozen ESM-2 head adaptation | Yes | Same strict budget | Tests whether representation updates are needed |

Frozen ESM-2 + ListMLE remains a descriptive reference. The head-only Reptile model is a secondary,
low-cost control; the primary paired contrast is:

```text
Reptile-initialized LoRA + strict-budget target adaptation
minus
standard-initialization LoRA + identical strict-budget target adaptation
```

The pooled and Reptile methods must use the same strict auxiliary cohort, admissible examples,
assay-sampling distribution, and matched training exposure. Total labeled-example exposures and
approximate backbone forward/backward token counts must be within 5% and will be recorded.

## 6. Exact Reptile objective

Reptile learns an initialization by adapting to a sampled task and moving the shared parameters
toward the task-adapted parameters. Let `phi` contain the LoRA and residual-head parameters. For an
auxiliary assay `tau`:

1. copy the current initialization `phi` to task parameters `phi_tau`;
2. take `K >= 2` inner gradient steps on `K` different within-assay ListMLE batches;
3. sample each inner batch from different residue positions where possible;
4. update the shared initialization using:

```text
phi <- phi + epsilon * (phi_tau - phi)
```

There is no query-loss gradient in the Reptile outer update. A query loss used directly for the
outer gradient would instead define a FOMAML-style experiment and must be named and tested
separately.

`K = 1` is prohibited because it collapses toward ordinary joint training and weakens the intended
within-task adaptation signal. The final `K`, inner learning rate, outer step size, and number of
episodes will be selected using post-adaptation performance on meta-validation proteins, then
frozen.

Reptile is the initial choice instead of full MAML because it avoids differentiating through the
complete inner trajectory and is practical for LoRA on a single paid-Colab GPU. Full MAML and
FOMAML are outside the initial confirmatory matrix.

## 7. Auxiliary-assay eligibility and leakage policy

Auxiliary assays will come from the authoritative ProteinGym v1.3 substitutions release already
used for TEM-1. Selection must use metadata and integrity criteria only--never target-test
performance.

### Basic eligibility

An assay must:

1. contain single amino-acid substitutions only;
2. contain at least 500 valid, unique single substitutions after integrity checks;
3. have a target sequence no longer than 1,022 residues;
4. provide a finite ProteinGym `DMS_score` with documented higher-is-better direction;
5. have mutation identifiers and mutated sequences consistent with its reference sequence;
6. have enough mutated positions for disjoint adaptation and evaluation sets;
7. have provenance and licensing recorded before training.

### Protein-level deduplication

All assays sharing a target sequence or UniProt protein form one protein group. A group can appear
in only one of meta-train, meta-validation, reserved meta-test, related-protein exploration, or
same-protein positive control.

For the strict cohort, one assay per protein is chosen by this outcome-blind rule:

1. greatest number of valid single substitutions;
2. greatest fraction of mutated positions covered;
3. lexicographically smallest `DMS_id` as the final tie-breaker.

Multiple phenotypes from one protein are not treated as independent tasks in the strict analysis.

### Cohort A -- sequence-dissimilar strict cohort (primary)

The strict cohort excludes:

- `BLAT_ECOLX_Stiffler_2015` itself;
- every assay with the same UniProt identifier or exact target sequence as TEM-1;
- the Deng, Firnberg, and Jacquier TEM-1 assays;
- a protein with at least 30% aligned identity to TEM-1 over at least 70% of the TEM-1 sequence;
- an assay annotated as TEM/beta-lactamase family even if an automated screen misses it.

This cohort will be described as **sequence-dissimilar under the pre-specified screen**, not as
provably nonhomologous. A 30% identity threshold cannot establish evolutionary unrelatedness.

### Cohort B -- biologically related but distinct proteins (exploratory)

A separate exploratory cohort may contain distinct beta-lactamases or functionally related enzymes
that fail the strict relatedness screen but:

- do not share the target UniProt identifier or exact sequence;
- do not use Stiffler target labels;
- pass all mutation and provenance integrity checks;
- are labeled explicitly as related-protein transfer rather than broad cross-protein transfer.

This cohort tests whether biological relatedness makes transfer more effective. It cannot replace
the strict cohort or rescue a failed primary result.

### Same-protein positive control

The three other exact-sequence TEM-1 assays may be retained only as a clearly labeled positive
control. They never enter the strict or related cross-protein cohorts and cannot support a
cross-protein claim.

### Task-level partition

Before training, eligible strict protein groups will be divided into:

- meta-train proteins, which provide gradients;
- meta-validation proteins, which select initialization checkpoints and hyperparameters;
- five reserved meta-test proteins, which remain untouched until external confirmation.

The partition will be stratified where practical by broad selection type and sequence-length bin.
If five suitable reserved protein groups cannot be formed, implementation pauses and the external-
confirmation design is revised before any training begins.

The exact registry, exclusion reason, identity/coverage result, biological annotation, partition,
and file checksum will be committed before model development.

## 8. Handling incompatible assay scales

Raw DMS scores are not comparable between experiments. Phase 3 will not pool raw target values or
fit a global score standardizer.

- ProteinGym's direction-normalized score establishes only that higher is better.
- ListMLE lists are constructed within one assay only.
- Lossless ordinal ranks are derived independently inside each assay from float64 source scores.
- A training batch or Reptile episode never mixes mutations from two assays in one ranking list.
- Spearman correlation is computed per assay before aggregation.

This preserves within-assay order without assuming that the numeric score scale has the same
biological meaning across experiments.

## 9. Model, meta-training, and meta-validation

### Shared mutation scorer

The starting architecture remains:

```text
prediction = fixed ESM-2 masked-marginal score
           + learned alternate-residue residual
           - learned wild-type-residue residual
```

The main pooled and Reptile models train rank-16 LoRA adapters and the 320-to-20 residual head. The
head-only control trains only the residual head with ESM-2 frozen. Public ESM-2 base weights remain
unchanged in every method.

### Reptile meta-training episode

1. Sample one meta-training protein task with equal probability per protein.
2. Reset task parameters to the current shared initialization.
3. Draw at least two inner batches from distinct residue positions in that assay.
4. Apply sequential ListMLE inner updates; later steps operate on the parameters produced by earlier
   steps.
5. Apply the Reptile parameter-difference outer update.
6. Record task identity, positions, losses, gradient norms, and update magnitude.

Training has no query-loss gradient. Disjoint held-out positions are used only in meta-validation.

### Pooled-pretraining control

The pooled control samples assays and within-assay ListMLE batches from the same distribution but
performs ordinary joint optimization without resetting to a shared initialization for task-specific
inner trajectories. It answers whether gains come from a meta-objective or simply more supervised
data.

### Adaptation-based meta-validation

Checkpoint and hyperparameter selection must measure **post-adaptation**, not merely pre-adaptation,
performance:

1. take one held-out meta-validation protein;
2. sample strict budgets of 20, 40, and 401 labels from adaptation positions;
3. adapt using the candidate fixed schedule;
4. evaluate on disjoint residue positions from the same protein;
5. repeat across subset seeds and all meta-validation proteins;
6. select using mean post-adaptation Spearman, with the 401-label metric primary and 20/40-label
   behavior reported as secondary.

No meta-validation example supplies a meta-training gradient. The chosen checkpoint must be the one
that becomes useful after a small amount of adaptation, not simply the one with the best unadapted
predictions.

## 10. Phase 3A -- TEM-1 extended evaluation

The former seeds 42--46 remain historical Phase 1/2 development evidence. Phase 3A uses new
position-partition seeds:

```text
142, 143, 144, 145, 146
```

These partitions reduce direct reuse of former test assignments but are still an extended
evaluation of the same biological dataset.

### Strict-budget split for each seed

1. Split whole TEM-1 positions into a 90% adaptation-candidate pool and 10% final test set.
2. Keep every substitution at a test position completely absent from adaptation.
3. Draw nested strict label budgets only from the adaptation-candidate positions.
4. Use subset-sampling seeds `11`, `12`, and `13` below the full candidate pool.
5. Give every comparator the identical mutation manifest at a budget.
6. Run the fixed adaptation schedule with no extra target validation labels and no target-specific
   early stopping.
7. Evaluate the resulting fixed-step checkpoint once on test positions after protocol lock.

The optimization seed is fixed to `1`, following Phase 2 evidence that position and subset choice
dominate optimization-seed variation.

### Strict total-label budgets

```text
20, 40, 200, 401, 1,002, 2,004, and the full adaptation-candidate pool
```

The 401-label outcome is primary. For the full-pool endpoint, the exact available count will vary
slightly with the held-out test positions and will be reported rather than forced to equal the old
80/10/10 count.

### Fair adaptation

Standard, pooled, Reptile, and head-only initializations must use the same:

- target subset manifest;
- ListMLE loss and exact ordinal targets;
- list size and effective lists per optimizer step;
- fixed number of adaptation updates;
- adaptation learning rates and checkpoint rule;
- one-time target-test evaluation policy.

Initialization and the explicitly declared trainable parameter set are the only intended
differences.

### Secondary fixed-validation diagnostic

If run, it uses the former 80/10/10 convention and reports both training-label and validation-label
counts. It is never labeled as the strict-budget result.

## 11. Phase 3B -- fresh-protein confirmation

Five protein groups will be reserved during the assay audit and remain unopened for method
selection. After every method and hyperparameter is locked from meta-train, meta-validation, and
TEM-1 development:

1. apply strict 20-, 40-, and 401-label budgets to each reserved protein;
2. use three deterministic position-held-out support/test partitions per protein;
3. give all comparators identical target mutation manifests;
4. use the same fixed adaptation schedule without target-specific validation labels;
5. average subset/split repeats within each protein;
6. treat the five protein-level means as the external-confirmation units.

The exact partition seeds will be generated and committed during the audit, before any model
training. TEM-1 outcomes may not change the Phase 3B model, schedule, cohorts, metrics, or success
rule.

## 12. Development and test firewall

Development has three layers:

1. **Technical development:** synthetic/tiny assay slices verify gradients, memory, determinism,
   resume, and artifact integrity without scientific evaluation.
2. **Auxiliary development:** meta-train supplies gradients; meta-validation post-adaptation metrics
   choose checkpoints and settings.
3. **Historical TEM-1 development:** former split 42 may diagnose compatibility with the existing
   pipeline, but cannot select a setting that will be described as fresh-protein evidence.

Before either target evaluation, a machine-readable protocol lock records:

- auxiliary file hashes and protein-task partitions;
- source-code commit;
- Reptile equation and resolved inner/outer settings;
- pooled exposure-matching settings;
- fixed adaptation schedule for every budget;
- target split and subset-manifest hashes;
- expected run identities and aggregation rules;
- permitted outputs and explicit test-opening state.

Scripts must reject configuration drift. Reserved-protein test metrics remain unavailable until the
full external-confirmation protocol is locked.

## 13. Evaluation and reporting

### Primary metric

Overall Spearman correlation across mutations at target positions absent from adaptation.

### Diagnostic metrics

- between-position Spearman;
- position-centered Spearman;
- pooled within-position pairwise ranking accuracy;
- macro per-position Spearman where defined;
- post-adaptation change from the unadapted initialization;
- selected auxiliary checkpoint, losses, gradient/update norms, runtime, and GPU memory;
- per-protein meta-validation and external-confirmation Spearman.

### TEM-1 aggregation

The five outer position splits are the units. Three nested subset samples are averaged within a
split before calculating mean and sample standard deviation across splits.

### Fresh-protein aggregation

Splits and subset samples are averaged within each reserved protein. The five protein-level means,
not individual mutations or repeated subsets, are the external-confirmation units.

For each method and budget, report:

- mean, sample standard deviation, minimum, and maximum across the relevant units;
- every split-level or protein-level value;
- paired differences against standard initialization, pooled pretraining, and zero-shot;
- win/tie/loss counts using a declared numerical tie tolerance;
- within-unit variation from splits/subset samples;
- the smallest strict budget meeting the practical `+0.02` threshold;
- every target label accessed outside final evaluation, which must equal the stated budget.

Failed runs, exclusions, and protocol deviations remain visible in the run registry. No metric is
inferred from validation behavior or entered without a generated result artifact.

## 14. Pre-specified decision rules

### TEM-1 extended-evaluation rule

The Phase 3A result is positive at 401 total labels only if:

1. Reptile-initialized LoRA exceeds standard-initialization LoRA by at least `0.02` mean Spearman;
2. the paired difference is positive on at least four of five outer splits;
3. mean position-centered Spearman is not lower by more than `0.02`;
4. label-accounting, completeness, leakage, and protocol-lock checks pass.

### Fresh-protein confirmation rule

Evidence for cross-protein few-shot transfer requires, at 401 total labels:

1. Reptile-initialized LoRA exceeds standard-initialization LoRA by at least `0.02` across the five
   protein-level means;
2. the paired difference is positive on at least four of five reserved proteins;
3. mean position-centered Spearman is not lower by more than `0.02`;
4. the entire reserved-protein matrix passes protocol validation.

### Interpretation of other outcomes

- **Meta and pooled both help similarly:** cross-assay supervision helps, but Reptile adds no clear
  value beyond ordinary pretraining.
- **Pooled beats meta:** transfer is useful, but the selected Reptile protocol is not the best
  optimization strategy.
- **Head-only matches LoRA:** most benefit comes from a transferable prediction head rather than
  representation adaptation.
- **Only the related cohort helps:** transfer depends on biological relatedness and does not show
  broad sequence-dissimilar transfer.
- **Only the same-protein control helps:** transfer is highly target-specific.
- **TEM-1 helps but reserved proteins do not:** report a TEM-1 case-study success, not generalizable
  meta-learning.
- **No method helps at low budgets:** the label threshold is not shifted by this cohort and model.
- **Mean improves without the required win count:** the result remains target/split-sensitive and
  the primary rule is not met.

## 15. Milestones and stopping points

### Milestone 0 -- revised design freeze (this document)

- Fix strict total-label accounting, exact Reptile semantics, cohort roles, adaptation-based
  meta-validation, TEM-1 claim limits, and fresh-protein confirmation.
- Stop before implementation.

### Milestone 1 -- authoritative assay audit

- Build the assay registry and verify provenance/licensing.
- Apply integrity, sequence-length, protein-deduplication, identity, and annotation rules.
- Identify strict, related, same-protein, meta-train, meta-validation, and five meta-test groups.
- Commit every inclusion/exclusion reason, partition, and checksum.
- Stop for review before a large download or preprocessing run.

### Milestone 2 -- data and zero-shot preparation

- Download only the frozen cohorts from the authoritative ProteinGym source.
- Validate mutation conventions, sequences, directionality, and licenses.
- Generate per-assay position partitions and masked-marginal zero-shot caches.
- Stop after a data-integrity report.

### Milestone 3 -- training infrastructure

- Add multi-assay loading, within-assay ranking batches, task partitions, pooled training, exact
  Reptile updates, head-only control, strict label ledgers, compact checkpoints, deterministic
  resume, and optional W&B mirroring.
- Test task leakage, `K >= 2`, distinct inner batches, no query gradient, task balance, ordinal
  targets, exposure matching, label accounting, checkpoint restoration, and config mismatch errors.
- Run no scientific experiment.

### Milestone 4 -- technical smoke test and CUDA benchmark

- Use tiny data slices and very few episodes.
- Verify gradient flow, update equations, memory, resume, and numerical stability.
- Benchmark microbatches on the actual Colab GPU.
- Open no new target-test metrics.

### Milestone 5 -- adaptation-based meta-validation

- Compare the predeclared small hyperparameter grid.
- Evaluate 20-, 40-, and 401-label post-adaptation performance on held-out meta-validation proteins.
- Match pooled and Reptile exposure.
- Freeze the auxiliary checkpoint and common target-adaptation schedule.

### Milestone 6 -- protocol lock

- Freeze source commit, assay/task manifests, settings, label ledgers, expected runs, metrics,
  aggregation, and test-opening rules.
- Save and verify the protocol checksum before Phase 3A or Phase 3B test evaluation.

### Milestone 7 -- Phase 3A TEM-1 extended evaluation

- Run strict-budget standard, pooled, Reptile, and head-only adaptation on seeds 142--146.
- Optionally run the separately labeled fixed-validation diagnostic.
- Analyze TEM-1 without modifying the locked Phase 3B protocol.

### Milestone 8 -- Phase 3B fresh-protein confirmation

- Run the locked method matrix on the five untouched protein groups.
- Validate completeness and label accounting before aggregation.
- Open each test partition once.

### Milestone 9 -- final analysis and project decision

- Produce strict-budget learning curves, paired tables, diagnostics, and compute accounting.
- Update `EXPERIMENT_LOG.md` with results and limitations.
- State separately whether the evidence supports TEM-1 transfer and cross-protein transfer.

## 16. Reproducibility and monitoring

Repository JSON/CSV artifacts and checkpoints remain authoritative. W&B may mirror training and
meta-validation curves, but a W&B outage must not alter or invalidate a run. Every run remains
resumable from Drive or local artifacts without W&B.

Random seeds, task order, subset identities, label ledgers, package versions, hardware, wall time,
peak memory, and source commit will be stored. Generated results stay under `results/` and are never
entered manually into documentation.

## 17. Related research and design context

- [MAML](https://proceedings.mlr.press/v70/finn17a.html) defines meta-learning around performance
  after a small number of adaptation steps.
- [Reptile](https://arxiv.org/abs/1803.02999) provides the first-order parameter-difference update
  used here.
- [ProteinGym](https://proceedings.neurips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf)
  supplies standardized DMS tasks.
- [FSFP](https://www.nature.com/articles/s41467-024-49798-6) combines protein language models,
  LoRA, ListMLE, and meta-transfer learning and reports that auxiliary-task size and similarity can
  affect transfer.
- [Metalic](https://proceedings.iclr.cc/paper_files/paper/2025/hash/798095a4827e2ce739b16cf23acc876c-Abstract-Conference.html)
  evaluates meta-learning across protein-fitness tasks and transfer to unseen tasks.

These works establish that the overall direction is relevant, but they do not predetermine the
result of this smaller controlled Reptile experiment.

## 18. Expected conclusion boundaries

If only the Phase 3A rule is met, the defensible conclusion is:

> Under a strict 401-label budget, a Reptile-initialized LoRA model improved adaptation to new
> TEM-1 position partitions relative to standard-initialization LoRA.

If both Phase 3A and Phase 3B rules are met, the stronger conclusion is:

> A LoRA initialization meta-learned from the pre-specified ProteinGym task cohort improved strict-
> budget adaptation on TEM-1 and on five previously untouched protein tasks under the locked
> evaluation protocol.

Neither result justifies claiming that meta-learning universally improves protein mutation-effect
prediction.
