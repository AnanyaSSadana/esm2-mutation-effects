# Phase 3A: one-day cross-assay meta-learning feasibility pilot

**Status:** active implementation plan. No Phase 3A auxiliary assay table has
yet been downloaded, no Phase 3A model has been trained, and no new TEM-1 test
score has been opened.

**Supersedes for the current deadline:** the full Phase 3 program is deferred
to [Phase 3B](PHASE3_META_LEARNING_DESIGN.md). Phase 3A must never be reported
as cross-protein confirmation or as a stable estimate across target splits.

## Purpose and claim boundary

Phase 3A is a bounded feasibility experiment:

> With a small, pre-specified, sequence-dissimilar ProteinGym auxiliary cohort,
> does a Reptile-initialized LoRA model improve strict-budget adaptation to one
> new held-out-position TEM-1 partition compared with standard target-only
> LoRA?

It can establish only whether this implementation is technically viable and
whether the selected target partition merits a larger experiment. A result from
one target split, one 401-label subset, and one auxiliary cohort cannot support
a general cross-protein-transfer claim, a statistical-significance claim, or
the former four-of-five-split decision rule.

## Locked pilot scope

### Auxiliary task shortlist

The audit starts from these five distinct, metadata-screened ProteinGym v1.3
assays. Their labels are never used until their raw tables pass the required
integrity checks.

| Role | DMS ID | Protein | Metadata rationale |
|---|---|---|---|
| Meta-train | `AMIE_PSEAE_Wrenbeck_2017` | Aliphatic amidase | 346 aa; 6,227 metadata single substitutions |
| Meta-train | `CALM1_HUMAN_Weile_2017` | Calmodulin-1 | 149 aa; 1,813 metadata single substitutions |
| Meta-train | `CCDB_ECOLI_Tripathi_2016` | CcdB toxin | 101 aa; 1,663 metadata single substitutions |
| Meta-train | `DYR_ECOLI_Nguyen_2023` | DHFR | 159 aa; 2,916 metadata single substitutions |
| Diagnostic only | `CASP3_HUMAN_Roychowdhury_2020` | Caspase-3 | 258 aa; 1,567 metadata single substitutions |

The metadata preflight excludes exact TEM-1 duplicates and explicit
beta-lactamase/TEM-family tasks from this strict pilot. If any shortlisted
table fails raw integrity, it is replaced by the first raw-integrity-passing
metadata candidate in lexicographic DMS-ID order that has a distinct protein
group and passes the same screen. Every replacement must be recorded before
training; it cannot depend on model performance.

### Target and label policy

- **Target assay:** `BLAT_ECOLX_Stiffler_2015` (TEM-1).
- **One target outer split:** position-held-out seed `142`, creating a 90%
  adaptation-candidate pool and 10% test-position pool.
- **One strict target budget:** exactly **401** mutation labels sampled from
  adaptation-candidate positions with subset seed `11`.
- **One optimization seed:** `1`.
- All 401 accessible target labels are used for adaptation. There is no target
  validation set, early stopping, learning-rate tuning, calibration, or model
  selection.
- The target test labels are opened exactly once, only after the source commit,
  raw-data manifest, target manifest, and run matrix are locked.

### Comparators

Every supervised method uses the same target manifest, rank-16 LoRA,
masked-marginal residual scoring, ListMLE lists, fixed number of target updates,
and fixed optimizer settings. The only intended difference is initialization.

1. Zero-shot ESM-2: no labels; fixed reference.
2. Standard initialization -> target-only LoRA adaptation.
3. Pooled multi-assay initialization -> target-only LoRA adaptation.
4. Reptile initialization -> target-only LoRA adaptation.

The head-only meta-learning control, multiple target budgets, repeated target
splits, and fresh-protein confirmation are deferred to Phase 3B.

### Fixed training schedule

The pilot uses Phase 2's model and optimizer settings without a hyperparameter
search:

```text
model: facebook/esm2_t6_8M_UR50D
LoRA rank / alpha / dropout: 16 / 16 / 0.05
ListMLE list size / lists per update: 5 / 16
microbatch: selected by a short CUDA safety benchmark (start at 16)
head LR / LoRA LR / weight decay: 1e-3 / 1e-4 / 0.01
gradient-norm clipping: 3.0
```

Reptile and pooled pretraining each receive exactly 300 within-assay ListMLE
optimizer updates. Reptile uses 150 episodes, two sequential inner updates per
episode (`K=2`), then applies the parameter-difference update
`phi <- phi + epsilon * (phi_task - phi)` with `epsilon = 0.1`. Pooled
pretraining samples the same four tasks with equal probability and performs 300
ordinary ListMLE updates. The runner records actual per-task update counts and
refuses an exposure mismatch greater than one update.

All three supervised target methods receive 50 fixed target-adaptation updates.
The 401-label target budget therefore refers only to target score access; it
does not count auxiliary supervision.

### Tied auxiliary scores

Several valid ProteinGym assays have repeated float64 `DMS_score` values. A
ListMLE list is therefore constructed by first selecting five distinct score
values within one assay and then one mutation uniformly from each selected
score-value bin. The five values are ranked using their unmodified float64
order. No arbitrary order is imposed within a tied score bin, and no ranking
list mixes assays. The data gate requires at least five distinct score values
per task and records their counts.

## Required gates

1. **Data gate:** download only the five shortlisted ProteinGym tables; validate
   single-substitution notation, mutated sequence, finite normalized score,
   one-based positions, coverage, provenance, and checksums.
2. **Technical gate:** run a tiny CPU/CUDA smoke test that checks Reptile's
   parameter-difference update, two distinct within-assay batches, resume, task
   balance, and no target-test evaluation.
3. **Protocol lock:** save raw-file and manifest hashes, code commit, fixed
   schedules, expected run identities, and an explicit `test_opened: false`
   state.
4. **Pilot evaluation:** run the four-model matrix once and open the target
   test subset exactly once.

## Reporting

The generated report will show overall Spearman, position-centered Spearman,
between-position Spearman, and within-position pairwise accuracy for the one
target test partition. It will show the raw Reptile-minus-standard and
Reptile-minus-pooled differences, all update counts, label ledger, run hashes,
and GPU runtime/memory.

The conclusion template is deliberately narrow:

> On one pre-specified TEM-1 held-out-position partition and a strict
> 401-label target budget, the Phase 3A pilot observed [direction and
> magnitude] for Reptile initialization relative to the controls. This is a
> feasibility result, not evidence of robust or cross-protein generalization.

## Deferred Phase 3B

The original complete program remains the future direction in
[PHASE3_META_LEARNING_DESIGN.md](PHASE3_META_LEARNING_DESIGN.md): larger
audited cohorts, adaptation-based meta-validation, repeated TEM-1 splits and
budgets, head-only control, matched pooled control, and five held-out fresh
protein groups. Phase 3A results cannot be used to alter the Phase 3B cohort,
hyperparameters, or decision rules after Phase 3B begins.
