# ESM-2 Mutation Effect Experiment Log

This log records completed runs for the TEM-1 `BLAT_ECOLX_Stiffler_2015` assay. Results are only added after the corresponding command has run successfully. Spearman's rho measures rank agreement between predicted and experimental mutation effects; larger values are better.

## Results through October 18, 2026

| Model | Training regime | Evaluation subset | N | Selected alpha | Validation rho | Evaluation rho |
|---|---|---:|---:|---:|---:|---:|
| ESM-2 masked marginal | Zero-shot | Full assay | 4,996 | — | — | 0.399278 |
| ESM-2 masked marginal | Zero-shot | Held-out test positions | 494 | — | — | 0.426226 |
| ESM-2 masked marginal | Zero-shot | Random held-out test variants | 500 | — | — | 0.372739 |
| Mutation-feature ridge | Supervised, position split | Held-out test positions | 494 | 100 | 0.401088 | 0.297027 |
| Mutation-feature ridge | Supervised, random split | Random held-out test variants | 500 | 10 | 0.408050 | 0.377767 |
| Frozen ESM-2 ridge | Supervised, position split | Held-out test positions | 494 | 1,000 | 0.662152 | 0.457278 |
| Frozen ESM-2 ridge | Supervised, random split | Random held-out test variants | 500 | 1 | 0.818370 | 0.812317 |

The zero-shot result was computed with `facebook/esm2_t6_8M_UR50D` on Apple MPS using a batch size of 16. It evaluates all 4,996 variants because zero-shot scoring does not train on assay labels.

The supervised results use validation performance to select the ridge regularization strength. The final model is then fitted on the combined training and validation subsets and evaluated once on the held-out test subset.

The frozen ESM-2 cache contains the wild-type representation, mutant representation, and their difference. The corrected linear model removes the exactly redundant mutant block and fits 640 float64 wild-type-plus-difference features with the `lsqr` ridge solver.

## Commands completed

```bash
python scripts/score_zero_shot.py --device auto --batch-size 16
python scripts/evaluate.py results/esm2_masked_marginal.csv
python scripts/evaluate.py results/esm2_masked_marginal.csv --split-strategy position --subset test
python scripts/evaluate.py results/esm2_masked_marginal.csv --split-strategy random --subset test
python scripts/train_mutation_baseline.py --split-strategy position
python scripts/train_mutation_baseline.py --split-strategy random
python scripts/extract_esm2_features.py --device auto --batch-size 16
python scripts/train_esm2_baseline.py --split-strategy position
python scripts/train_esm2_baseline.py --split-strategy random
```

Generated files:

- `results/esm2_masked_marginal.csv`
- `results/esm2_masked_marginal_all.json`
- `results/esm2_masked_marginal_position_test.json`
- `results/esm2_masked_marginal_random_test.json`
- `results/mutation_features_position.csv`
- `results/mutation_features_position.json`
- `results/mutation_features_random.csv`
- `results/mutation_features_random.json`
- `data/processed/esm2_t6_site_features.npz`
- `results/frozen_esm2_position.csv`
- `results/frozen_esm2_position.json`
- `results/frozen_esm2_random.csv`
- `results/frozen_esm2_random.json`

## Current interpretation

The simple mutation-feature baseline has meaningful predictive signal, even though it only uses the wild-type amino acid, alternate amino acid, and normalized position. Its test correlation falls from `0.377767` on the random split to `0.297027` on the position-held-out split, a decrease of about `0.081`. This confirms that prediction at unseen residue positions is substantially harder and supports using the position split as the primary supervised evaluation.

On held-out positions, zero-shot ESM-2 reaches `0.426226`: substantially above the supervised mutation-feature baseline (`0.297027`) and only about `0.031` below the supervised frozen ESM-2 model (`0.457278`). This is strong evidence that the pretrained model carries useful mutation-effect signal that transfers to residue positions unseen by the supervised model.

On the random test split, zero-shot ESM-2 (`0.372739`) and the mutation-feature baseline (`0.377767`) perform similarly, while supervised frozen ESM-2 is much stronger (`0.812317`). This large gain is specific to the setting where supervised training includes other substitutions at the same residue positions.

Frozen ESM-2 representations substantially outperform the mutation-feature baseline. On held-out positions, test correlation rises from `0.297027` to `0.457278`, an absolute improvement of about `0.160`. On the random split, it rises from `0.377767` to `0.812317`, an improvement of about `0.435`.

The frozen model's much larger drop from the random split (`0.812317`) to the position split (`0.457278`) shows why the split design matters. Random splitting lets the model train on other substitutions at positions that also occur in the test set. The position split asks it to extrapolate to completely unseen residue sites and is therefore the more credible primary result.

The wider alpha search confirms an interior optimum of `1,000` for the position split: validation correlation is `0.662152` at `1,000`, compared with `0.658720` at `10,000` and `0.639512` at `100,000`. The random split retains an interior optimum at alpha `1`.

The corrected runs emitted no conditioning warning. They supersede the earlier provisional frozen results, which used all 960 cached dimensions even though `difference = mutant - wild type`, and therefore supplied an exactly redundant design matrix to the linear model. Removing that redundancy and using float64 `lsqr` fitting increased the position-held-out test correlation from `0.416847` to `0.457278`; the random-split result remained effectively unchanged.

## Repeated position split robustness

The three baselines were evaluated across five deterministic 80/10/10 position splits. Every substitution at a residue remained in the same subset, supervised alpha selection used only validation positions, and the checked-in split manifest was not modified.

```bash
python scripts/run_repeated_position_splits.py --seeds 42 43 44 45 46
```

### Per-seed test Spearman correlation

| Seed | Mutation features | Zero-shot ESM-2 | Frozen ESM-2 |
|---:|---:|---:|---:|
| 42 | 0.297027 | 0.426226 | 0.457278 |
| 43 | 0.303259 | 0.656576 | 0.583010 |
| 44 | 0.468794 | 0.342814 | 0.434136 |
| 45 | 0.092522 | 0.471889 | 0.204474 |
| 46 | 0.324590 | 0.291610 | 0.553376 |

### Aggregate test Spearman correlation

| Model | Mean | Sample standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Mutation features | 0.297238 | 0.134280 | 0.092522 | 0.468794 |
| Zero-shot ESM-2 | 0.437823 | 0.141026 | 0.291610 | 0.656576 |
| Frozen ESM-2 | 0.446455 | 0.149083 | 0.204474 | 0.583010 |

The seed-42 frozen result (`0.457278`) is close to the five-seed mean (`0.446455`), so it was not an unusually favorable estimate. However, the wide range from `0.204474` to `0.583010` shows that performance depends strongly on which residue positions are held out. Frozen ESM-2 has only a small mean advantage over zero-shot ESM-2 across these seeds, and it does not win on every split. Future claims should therefore report the repeated-split distribution rather than a single position split.

Generated robustness files:

- `results/repeated_position_splits/per_seed.csv`
- `results/repeated_position_splits/summary.csv`
- `results/repeated_position_splits/summary.json`

## Next verification steps

1. Run sample-efficiency experiments using progressively smaller labeled training subsets.
2. Consider expanding robustness evaluation beyond five seeds or using nested position-level cross-validation.
