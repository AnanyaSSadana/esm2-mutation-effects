# Phase 3B-lite: repeated target-split replication

**Status:** pre-registered implementation. This is a bounded follow-up to the completed Phase 3A
feasibility pilot, not the full Phase 3B program described in
[PHASE3_META_LEARNING_DESIGN.md](PHASE3_META_LEARNING_DESIGN.md).

## Question

Phase 3A's one strict 401-label TEM-1 split found that pooled auxiliary-assay initialization
improved overall Spearman over target-only LoRA, while Reptile did not improve beyond pooled
initialization. Phase 3B-lite asks whether that pattern replicates over additional, pre-specified
held-out residue-position partitions of the same TEM-1 assay.

## Locked design

- **Historical Phase 3A result:** split seed 142, already evaluated once and never reopened.
- **New target splits:** seeds 143, 144, 145, and 146.
- **Label policy:** position-held-out 90/10 partitions; exactly 401 accessible adaptation labels
  per split, sampled with subset seed 11. Every test-position label remains inaccessible during
  adaptation.
- **Methods:** zero-shot ESM-2, standard target-only LoRA, pooled multi-assay initialization plus
  LoRA, and first-order Reptile initialization plus LoRA.
- **Unchanged training schedule:** ESM-2 t6 8M, rank-16 LoRA, ListMLE lists of five mutations,
  16 lists per update, 50 target-adaptation updates, and optimization seed 1. Pooled and Reptile
  each receive the same 300 auxiliary-assay updates as Phase 3A.
- **Auxiliary cohort:** the same four Phase 3A meta-training tasks. No task, hyperparameter, or
  schedule is selected using new target-test results.
- **Efficiency without changed supervision:** pooled and Reptile initializations are trained once
  from the fixed auxiliary cohort and reused for every new target split. They never consume TEM-1
  labels, so this is computational reuse rather than information sharing across target tests.

The protocol lock and all four manifests are created before any new target test metric is
evaluated. A per-split runner refuses to overwrite existing test predictions; a completed split is
never re-run.

## Primary and secondary analyses

The primary analysis is the paired five-split distribution of **pooled minus standard overall
Spearman**, combining the existing Phase 3A seed-142 result with the four newly evaluated splits.
Report its mean, sample standard deviation, range, and wins/ties/losses. Reptile-versus-standard,
Reptile-versus-pooled, and comparisons to zero-shot are secondary. Report between-position and
within-position metrics descriptively, but do not select a winner using them.

This is still one target protein and one fixed target-label budget. It can support a stronger
within-TEM-1 replication claim than Phase 3A alone, but it cannot establish cross-protein
generalization or tune a meta-learning algorithm. Those remain Phase 3B future work.
