# ESM-2 mutation-effect prediction

A reproducible starter repository for predicting experimentally measured effects of single amino-acid substitutions with the small `facebook/esm2_t6_8M_UR50D` protein language model.

The selected assay is **BLAT_ECOLX_Stiffler_2015**, a nearly exhaustive deep mutational scan of TEM-1 beta-lactamase under stringent ampicillin selection. It has 4,996 unique single substitutions at 263 positions (24–286) in a 286-residue reference sequence. This is a strong first assay because it is well documented, has broad mutation coverage, contains no multi-mutants, and fits comfortably below ESM-2's 1,022-residue input limit.

Generated prediction files and feature caches are not checked into this repository. Running the commands below creates them under `results/` and `data/processed/`; compact observed results are recorded separately.

Completed experiment results and interpretations are recorded in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
The frozen protocol for the next controlled LoRA and ranking phase is recorded in
[`docs/LORA_EXPERIMENT_DESIGN.md`](docs/LORA_EXPERIMENT_DESIGN.md).
The Phase 2 label-efficiency protocol is recorded in
[`docs/PHASE2_LABEL_EFFICIENCY_DESIGN.md`](docs/PHASE2_LABEL_EFFICIENCY_DESIGN.md).

## What is included

- The processed ProteinGym v1.3 assay and its full reference metadata.
- Dataset provenance, checksums, score direction, mutation conventions, and licensing notes.
- Biological-integrity validation for every mutant and full mutated sequence.
- Reproducible 80/10/10 random and position-held-out train/validation/test splits.
- ESM-2 masked-marginal zero-shot scoring.
- A supervised ridge baseline using mutation identity and position.
- A supervised ridge baseline using frozen ESM-2 mutation-site representations.
- Spearman rank-correlation evaluation.

## Setup

Python 3.9–3.13 is supported. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The first ESM-2 command downloads the public model weights and caches them through PyTorch. No Hugging Face token is needed for the fair-esm checkpoint.

## Verify the data and splits

```bash
python scripts/validate_dataset.py
python scripts/make_splits.py
python -m unittest discover -s tests -v
```

The checked-in split manifest uses seed 42. Regenerating it with the default command is deterministic.

### Why two split strategies?

- `random` tests interpolation when variants at a residue may appear in both train and test.
- `position` keeps every substitution at a residue in the same subset. This is harder and better measures generalization to unseen positions.

Use validation only for model selection. Treat the test subset as a one-time final comparison. The supervised scripts default to the stricter `position` split.

## Run the baselines

### 1. Zero-shot masked-marginal ESM-2

```bash
python scripts/score_zero_shot.py --device auto --batch-size 16
python scripts/evaluate.py results/esm2_masked_marginal.csv
```

For apples-to-apples comparisons with the supervised test results, evaluate the same subsets:

```bash
python scripts/evaluate.py results/esm2_masked_marginal.csv \
  --split-strategy position --subset test
python scripts/evaluate.py results/esm2_masked_marginal.csv \
  --split-strategy random --subset test
```

For mutation `W165A`, the score is:

```text
log p(A at 165 | residue 165 masked) - log p(W at 165 | residue 165 masked)
```

Higher scores mean ESM-2 considers the mutant residue more plausible relative to the wild type. One masked forward pass is reused for all alternate residues observed at a position.

### 2. Basic supervised mutation-feature baseline

```bash
python scripts/train_mutation_baseline.py --split-strategy position
python scripts/train_mutation_baseline.py --split-strategy random
```

This ridge model uses one-hot wild-type and alternate residues plus normalized sequence position. It is deliberately simple and provides a useful non-PLM reference.

### 3. Frozen ESM-2 representation baseline

```bash
python scripts/extract_esm2_features.py --device auto --batch-size 16
python scripts/train_esm2_baseline.py --split-strategy position
python scripts/train_esm2_baseline.py --split-strategy random
```

For each mutant, the feature extractor caches the frozen layer-6 representation at the mutation site for the wild type, the mutant, and their difference. The linear baseline uses the wild-type and difference blocks (640 float64 features); it omits the mutant block because `mutant = wild type + difference` would make the design matrix exactly redundant. Ridge uses the numerically stable `lsqr` solver. Features are cached once in `data/processed/`; only the ridge regressor is trained.

Apple Silicon users can normally keep `--device auto` (MPS is selected when available). Reduce `--batch-size` if memory is limited. CPU works but frozen-feature extraction is the longest step because it embeds all 4,996 mutant sequences.

### 4. Repeated position-held-out robustness

After generating the zero-shot predictions and frozen feature cache, compare all three models across five deterministic position splits:

```bash
python scripts/run_repeated_position_splits.py --seeds 42 43 44 45 46
```

This keeps all substitutions at a residue in the same subset for every seed, selects each supervised model's alpha on that seed's validation positions, and evaluates once on its test positions. Outputs under `results/repeated_position_splits/` include per-seed results plus mean, sample standard deviation, minimum, and maximum test Spearman correlation. The checked-in `data/splits/splits.csv` is not modified.

### 5. Sample efficiency

Measure how the supervised models improve as labeled training mutations increase:

```bash
python scripts/run_sample_efficiency.py \
  --seeds 42 43 44 45 46 \
  --fractions 0.01 0.05 0.10 0.25 0.50 1.0
```

For each seed, the script uses the same position-held-out 80/10/10 split as the robustness experiment and constructs nested random subsets of the available training mutation rows. Validation positions remain fixed and are used only to select ridge alpha; they are not added to the final training fit. Test positions remain untouched. Zero-shot ESM-2 uses no labels and is repeated as a fixed reference at every fraction.

Outputs under `results/sample_efficiency/` include one row per seed, fraction, and model plus aggregate mean, sample standard deviation, minimum, and maximum test Spearman correlation. Fractions describe individual labeled mutation rows, not whole training positions.

Plot the aggregate learning curves with mean and one standard deviation across position splits:

```bash
python scripts/plot_sample_efficiency.py
```

This writes `results/sample_efficiency/learning_curve.png` and a vector PDF version. Run the sample-efficiency experiment first so that `summary.csv` exists.

### 6. Repeated training-subset sample efficiency

To determine whether the learning-curve shape depends on one fortunate training subset, repeat each fraction with independent subset samples inside every position split:

```bash
python scripts/run_repeated_sample_efficiency.py \
  --split-seeds 42 43 44 45 46 \
  --sampling-seeds 1 2 3 4 5 \
  --fractions 0.01 0.05 0.10 0.25 0.50 1.0
```

This produces 25 supervised runs per model for every fraction below 100%: five held-out-position splits times five nested training-subset samples. Full training is evaluated once per position split because every sampling seed selects the identical complete training set. The summary separately reports the overall standard deviation, variation between position-split means, and the mean sampling standard deviation within a fixed position split. Zero-shot ESM-2 is also evaluated once per position split rather than duplicated across sampling seeds.

Plot the repeated experiment with the existing plotting script:

```bash
python scripts/plot_sample_efficiency.py \
  --summary results/repeated_sample_efficiency/summary.csv \
  --output results/repeated_sample_efficiency/learning_curve.png \
  --pdf-output results/repeated_sample_efficiency/learning_curve.pdf
```

### 7. Representation and position diagnostics

Test whether the supervised models primarily learn position sensitivity or the relative effects of substitutions within a position:

```bash
python scripts/run_diagnostic_ablations.py --seeds 42 43 44 45 46
```

The experiment runs both random-variant and position-held-out splits. It compares the position mean, simple mutation features, categorical position plus mutation identity, zero-shot ESM-2, frozen wild-type-site features, frozen mutation-difference features, and their combined representation. Alongside overall Spearman correlation, it reports between-position Spearman, pooled within-position pairwise ranking accuracy, position-centered Spearman, and macro per-position Spearman where defined. Constant predictions are recorded as undefined rather than converted to zero.

Outputs are written under `results/diagnostic_ablations/`, including all test predictions so position-level failures can be inspected later.

### 8. Controlled masked-residual adaptation

The frozen protocol for the LoRA and ranking experiment is in
[`docs/LORA_EXPERIMENT_DESIGN.md`](docs/LORA_EXPERIMENT_DESIGN.md). The single-run training
interface supports the four controlled configurations:

```bash
python scripts/train_masked_adaptation.py --backbone frozen --loss mse
python scripts/train_masked_adaptation.py --backbone frozen --loss listmle
python scripts/train_masked_adaptation.py --backbone lora --loss mse
python scripts/train_masked_adaptation.py --backbone lora --loss listmle
```

These commands select the best checkpoint using validation positions and deliberately do **not**
evaluate test positions. They save the resolved configuration, latest and best checkpoints,
complete history, validation predictions, validation metrics, and a run summary beneath
`results/masked_adaptation/`. An interrupted run can continue with the identical command plus
`--resume`. Configuration mismatches and accidental reuse of an existing run directory fail
instead of overwriting artifacts.

Checkpoints contain only trainable parameters; the unchanged public ESM-2 backbone is restored
from its named checkpoint. This keeps the 190-run Phase 2 artifact set practical to persist while
retaining resume support and compatibility with older full-state checkpoints.

Test evaluation requires the explicit `--evaluate-test` option and should not be used until the
protocol reaches its designated test-evaluation milestone.

Run the approved five-epoch validation-only technical pilot with:

```bash
python scripts/run_lora_pilot.py --device auto
```

The pilot runs Frozen+MSE, Frozen+ListMLE, LoRA+MSE, and LoRA+ListMLE sequentially on split 42
with training seed 1. It streams epoch losses, validation Spearman, runtime, memory, and gradient
diagnostics; safely resumes existing partial runs; and writes
`results/lora_pilot/pilot_summary.json`. The pilot runner has no test-evaluation option.

The current pre-Phase-2 LoRA ListMLE microbatch benchmark can be run with:

```bash
python scripts/benchmark_lora_list_batching.py --device auto
```

This runs one validation-only epoch at microbatch sizes 4, 8, and 16 while retaining the effective
batch of 16 lists. Every candidate uses the identical 2,004-label Phase 2 manifest and the
lossless ordinal ranking targets. It continues if a larger size fails, records a technical
benchmark summary under `results/phase2_microbatch_benchmark/`, and cannot evaluate test
positions. Frozen-backbone runs automatically cache their fixed masked-site representations and
reuse them throughout training.

The direct post-correction MPS benchmark selected microbatch 4: one epoch took 129.57 seconds,
versus 163.63 seconds for size 8 and 4,104.25 seconds for size 16. Validation Spearman differed by
at most 0.000031 across candidates. Microbatch 4 therefore remains the local Phase 2 setting.
These checks did not evaluate test positions and are not scientific results; CUDA should be
benchmarked independently because its optimal batching can differ from MPS.

For a Google Colab run, use the upload-ready bundle and notebooks in [`colab/`](colab/). Notebook
01 selects the fastest safe CUDA microbatch without touching test positions. Notebook 02 reads
that choice, locks the protocol, runs or resumes the full Phase 2 matrix, and stores all results in
Google Drive.

Run the fixed Milestone 5 development matrix one optimization seed at a time with:

```bash
python scripts/run_lora_development.py --device auto --training-seed 1
python scripts/run_lora_development.py --device auto --training-seed 2
python scripts/run_lora_development.py --device auto --training-seed 3
```

Each command runs Frozen+MSE, Frozen+ListMLE, LoRA+MSE, and LoRA+ListMLE on position split 42.
The runner fixes the 100-epoch maximum and amended patience of 5, uses microbatch 4 with an
effective ListMLE batch of 16 lists, and has no test-evaluation option. Re-run the same command
after an interruption: incomplete runs resume from their latest checkpoint and completed runs are
skipped. Progress and validation-only aggregates are saved to
`results/lora_development/development_summary.json` after every completed run.

To launch all 12 runs sequentially instead, omit `--training-seed`. This can take many hours on a
laptop because LoRA+ListMLE is the expensive arm; the one-seed-at-a-time commands are easier to
manage.

After all 12 validation-only development runs complete, freeze the exact confirmatory protocol:

```bash
python scripts/lock_lora_protocol.py
```

The lock records the development-summary checksum, all model and optimization settings, the Git
commit, and a checksum of the confirmatory protocol. The confirmatory runner refuses an incomplete
or modified lock. It reuses the split-42 development checkpoints and evaluates each saved best
checkpoint once; splits 43–46 are trained under the same locked settings. The full matrix is run
with:

```bash
python scripts/run_lora_confirmatory.py --device auto
```

Partial split or optimization-seed selections are available for recoverable long runs. Completed
test runs are skipped, and the cumulative result is stored at
`results/lora_confirmatory/confirmatory_summary.json`.

The full Phase 1 pipeline can also be resumed with:

```bash
python scripts/run_phase1.py --device auto
```

Phase 1 is complete. Across five held-out-position splits, LoRA + ListMLE was the strongest
controlled method (mean test Spearman `0.473096 ± 0.113617` across split means), ahead of
zero-shot ESM-2 (`0.437823 ± 0.141026`) and Frozen + ListMLE (`0.435714 ± 0.132173`). It beat
zero-shot on three splits, tied on one, and lost on one, so it did not meet the predeclared
four-of-five split win criterion. See [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) for the complete
interpretation and [`docs/LORA_EXPERIMENT_DESIGN.md`](docs/LORA_EXPERIMENT_DESIGN.md) for the
locked protocol and result table.

### 9. Phase 2: label-efficiency confirmation

Phase 2 measures the number of labelled mutations required for Frozen + ListMLE and LoRA +
ListMLE to improve on zero-shot ESM-2 at unseen residue positions. It uses the same five outer
position splits, nested individual-mutation subsets of 20, 40, 200, 401, 1,002, 2,004, and 4,008
labels, and three sampling seeds for every budget below 100%. The full protocol and its leakage
safeguards are in [`docs/PHASE2_LABEL_EFFICIENCY_DESIGN.md`](docs/PHASE2_LABEL_EFFICIENCY_DESIGN.md).

The single resumable command runs a validation-only 20-label smoke check, freezes the protocol,
and then runs the 190 supervised confirmation jobs:

```bash
python scripts/run_phase2.py --device auto
```

For staged execution, run the smoke check and lock separately before the confirmation matrix:

```bash
python scripts/run_phase2_smoke.py --device auto
python scripts/lock_phase2_protocol.py
python scripts/run_phase2_label_efficiency.py --device auto
```

Completed jobs are skipped, interrupted jobs resume only with the same fingerprinted subset
manifest, and outputs are stored under `results/phase2_development/` and
`results/phase2_label_efficiency/`. Frozen-backbone jobs also share a checksummed,
device-qualified masked-site representation cache under `data/processed/`. The first frozen job
creates it and subsequent frozen jobs load the exact saved tensor instead of repeating 263 ESM-2
forward passes. LoRA jobs never use this cache because their backbone representations change
during training.

## Repository layout

```text
data/
  provenance/   authoritative metadata and source record
  raw/          downloaded processed assay (immutable)
  splits/       deterministic split assignments
  processed/    generated ESM-2 feature cache (gitignored)
scripts/        runnable validation, scoring, training, evaluation
src/            shared Python package
tests/          data and split integrity tests
results/        generated predictions/metrics (gitignored)
```

## Assay semantics

ProteinGym mutation strings are one-letter amino acids with **one-based** reference positions: `H24C` means histidine at reference position 24 is replaced by cysteine. ProteinGym's `DMS_score` has already been direction-normalized: **larger values always mean better measured fitness**. For this assay, the underlying phenotype is growth under 2,500 µg/mL ampicillin and the raw directionality is already positive (`+1`).

The full reference sequence includes the 23-residue signal peptide, while the assayed substitutions span residues 24–286. Do not subtract 23 from the mutation positions: the supplied mutation strings and full sequences already use full-length one-based coordinates.

See [DATASET_CARD.md](DATASET_CARD.md) and [data/provenance/SOURCE.json](data/provenance/SOURCE.json) before redistributing data.

## Sources and citation

- [ProteinGym v1.3 Zenodo record](https://doi.org/10.5281/zenodo.15293562)
- [ProteinGym repository](https://github.com/OATML-Markslab/ProteinGym)
- [ProteinGym benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html)
- [Stiffler et al. 2015](https://doi.org/10.1016/j.cell.2015.01.035)
- [Meta ESM repository](https://github.com/facebookresearch/esm)

The code in this repository is MIT-licensed. The ProteinGym Zenodo v1.3 record reports `mit-license`; attribution and the original-study citation are retained here and in `CITATION.cff`.
