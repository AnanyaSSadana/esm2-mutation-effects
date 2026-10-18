# ESM-2 Mutation Effect Experiment Log

This log records completed runs for the TEM-1 `BLAT_ECOLX_Stiffler_2015` assay. Results are only added after the corresponding command has run successfully. Spearman's rho measures rank agreement between predicted and experimental mutation effects; larger values are better.

## Results through October 18, 2026

| Model | Training regime | Evaluation subset | N | Selected alpha | Validation rho | Evaluation rho |
|---|---|---:|---:|---:|---:|---:|
| ESM-2 masked marginal | Zero-shot | Full assay | 4,996 | — | — | 0.399278 |
| Mutation-feature ridge | Supervised, position split | Held-out test positions | 494 | 100 | 0.401088 | 0.297027 |
| Mutation-feature ridge | Supervised, random split | Random held-out test variants | 500 | 10 | 0.408050 | 0.377767 |

The zero-shot result was computed with `facebook/esm2_t6_8M_UR50D` on Apple MPS using a batch size of 16. It evaluates all 4,996 variants because zero-shot scoring does not train on assay labels.

The supervised results use validation performance to select the ridge regularization strength. The final model is then fitted on the combined training and validation subsets and evaluated once on the held-out test subset.

## Commands completed

```bash
python scripts/score_zero_shot.py --device auto --batch-size 16
python scripts/evaluate.py results/esm2_masked_marginal.csv
python scripts/train_mutation_baseline.py --split-strategy position
python scripts/train_mutation_baseline.py --split-strategy random
```

Generated files:

- `results/esm2_masked_marginal.csv`
- `results/mutation_features_position.csv`
- `results/mutation_features_position.json`
- `results/mutation_features_random.csv`
- `results/mutation_features_random.json`

## Current interpretation

The simple mutation-feature baseline has meaningful predictive signal, even though it only uses the wild-type amino acid, alternate amino acid, and normalized position. Its test correlation falls from `0.377767` on the random split to `0.297027` on the position-held-out split, a decrease of about `0.081`. This confirms that prediction at unseen residue positions is substantially harder and supports using the position split as the primary supervised evaluation.

The zero-shot ESM-2 correlation of `0.399278` is encouraging, but it is not yet directly comparable with either supervised test result because it was evaluated on the complete assay. The next evaluation update should calculate the zero-shot correlation on the exact random and position-held-out test subsets.

## Next run

Extract frozen ESM-2 mutation-site representations:

```bash
python scripts/extract_esm2_features.py --device auto --batch-size 16
```

Expected output:

```text
data/processed/esm2_t6_site_features.npz
```

Then train and evaluate the frozen-representation ridge model:

```bash
python scripts/train_esm2_baseline.py --split-strategy position
python scripts/train_esm2_baseline.py --split-strategy random
```

The next update to this log should add both frozen ESM-2 test correlations, their selected alpha values, and the corresponding validation correlations.
