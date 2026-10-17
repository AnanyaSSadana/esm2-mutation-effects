# ESM-2 mutation-effect prediction

A reproducible starter repository for predicting experimentally measured effects of single amino-acid substitutions with the small `facebook/esm2_t6_8M_UR50D` protein language model.

The selected assay is **BLAT_ECOLX_Stiffler_2015**, a nearly exhaustive deep mutational scan of TEM-1 beta-lactamase under stringent ampicillin selection. It has 4,996 unique single substitutions at 263 positions (24–286) in a 286-residue reference sequence. This is a strong first assay because it is well documented, has broad mutation coverage, contains no multi-mutants, and fits comfortably below ESM-2's 1,022-residue input limit.

No model results are checked into this repository. Running the commands below creates them under `results/`.

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

Python 3.9–3.12 is supported. From this folder:

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

For each mutant, the feature extractor concatenates the frozen layer-6 representation at the mutation site for the wild type, the mutant, and their difference. Features are cached once in `data/processed/`; only the ridge regressor is trained.

Apple Silicon users can normally keep `--device auto` (MPS is selected when available). Reduce `--batch-size` if memory is limited. CPU works but frozen-feature extraction is the longest step because it embeds all 4,996 mutant sequences.

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
