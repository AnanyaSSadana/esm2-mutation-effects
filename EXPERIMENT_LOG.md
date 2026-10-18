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
python scripts/run_repeated_position_splits.py --seeds 42 43 44 45 46
python scripts/run_sample_efficiency.py --seeds 42 43 44 45 46 --fractions 0.01 0.05 0.10 0.25 0.50 1.0
python scripts/plot_sample_efficiency.py
python scripts/run_repeated_sample_efficiency.py --split-seeds 42 43 44 45 46 --sampling-seeds 1 2 3 4 5 --fractions 0.01 0.05 0.10 0.25 0.50 1.0
python scripts/plot_sample_efficiency.py --summary results/repeated_sample_efficiency/summary.csv --output results/repeated_sample_efficiency/learning_curve.png --pdf-output results/repeated_sample_efficiency/learning_curve.pdf
python scripts/run_diagnostic_ablations.py --seeds 42 43 44 45 46
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

## Position-held-out sample efficiency

The two supervised baselines were trained with nested random subsets of the available training mutation rows at each seed. Validation and test positions stayed fixed. Validation labels selected alpha but were not added to the final fit, so each percentage represents the number of training labels actually fitted. The same zero-shot test score is shown at every fraction as a no-label reference. The fixed validation set is an additional label overhead and is not included in the displayed training-label count.

### Aggregate test Spearman correlation across five seeds

| Training fraction | Mean fitted labels | Mutation features | Zero-shot ESM-2 | Frozen ESM-2 |
|---:|---:|---:|---:|---:|
| 1% | 40 | 0.047906 | 0.437823 | 0.268134 |
| 5% | 200 | 0.238683 | 0.437823 | 0.424249 |
| 10% | 401 | 0.252384 | 0.437823 | 0.461587 |
| 25% | 1,002 | 0.272699 | 0.437823 | 0.456946 |
| 50% | 2,004 | 0.284298 | 0.437823 | 0.446210 |
| 100% | 4,008 | 0.284438 | 0.437823 | 0.441424 |

Frozen ESM-2 reaches its highest five-seed mean in this run with 10% of the available training labels (`0.461587`), slightly above zero-shot (`0.437823`). More labels do not produce a monotonic increase, and split-to-split variation remains substantial. This suggests the frozen ridge baseline becomes useful quickly but is limited more by position shift, linear-probe capacity, and validation-based regularization than by raw label count alone.

The 100% results are not expected to exactly match the repeated-split robustness table. That earlier experiment refits supervised models on training plus validation rows after alpha selection. The sample-efficiency experiment fits only the requested fraction of training rows and never folds validation labels into the final model, preserving the training-label budget.

Generated sample-efficiency files:

- `results/sample_efficiency/per_seed_fraction.csv`
- `results/sample_efficiency/summary.csv`
- `results/sample_efficiency/summary.json`
- `results/sample_efficiency/learning_curve.png`
- `results/sample_efficiency/learning_curve.pdf`

## Repeated training-subset sample efficiency

The sample-efficiency experiment was repeated with five independent nested training-subset samples inside each of the five outer position splits. Fractions below 100% therefore have 25 supervised runs per model. Full training has one run per position split because every sampling seed selects the same complete training set. Zero-shot also has one result per position split.

### Mean test Spearman correlation and variation

| Training fraction | Fitted labels | Mutation features mean ± SD | Zero-shot mean ± SD | Frozen ESM-2 mean ± SD | Frozen within-split sampling SD | Frozen between-split SD of means |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 40 | 0.074228 ± 0.154706 | 0.437823 ± 0.141026 | 0.278312 ± 0.157968 | 0.130479 | 0.092359 |
| 5% | 200 | 0.190723 ± 0.157520 | 0.437823 ± 0.141026 | 0.393847 ± 0.124239 | 0.068386 | 0.110938 |
| 10% | 401 | 0.230105 ± 0.162546 | 0.437823 ± 0.141026 | 0.407220 ± 0.128987 | 0.075947 | 0.104142 |
| 25% | 1,002 | 0.261780 ± 0.146574 | 0.437823 ± 0.141026 | 0.428886 ± 0.128361 | 0.054920 | 0.123196 |
| 50% | 2,004 | 0.278757 ± 0.142539 | 0.437823 ± 0.141026 | 0.432453 ± 0.112901 | 0.038092 | 0.116234 |
| 100% | 4,008 | 0.284438 ± 0.157898 | 0.437823 ± 0.141026 | 0.441424 ± 0.130679 | 0.000000 | 0.130679 |

The apparent frozen ESM-2 peak at 10% in the single-subset experiment does not survive repeated subset sampling. The repeated curve instead shows a gradual improvement as labels increase. Frozen ESM-2 approaches zero-shot by 25–50% of the training labels and exceeds its five-split mean only narrowly at 100% (`0.441424` versus `0.437823`).

Training-subset selection matters most at very low label counts: the frozen model's mean within-split sampling standard deviation falls from `0.130479` at 1% to `0.038092` at 50%. At 25% and above, variation between held-out position splits is larger than variation from the sampled training rows. This reinforces the conclusion that which residue positions are held out is the primary source of uncertainty for this assay.

Generated repeated sample-efficiency files:

- `results/repeated_sample_efficiency/per_run.csv`
- `results/repeated_sample_efficiency/summary.csv`
- `results/repeated_sample_efficiency/summary.json`
- `results/repeated_sample_efficiency/learning_curve.png`
- `results/repeated_sample_efficiency/learning_curve.pdf`

## Representation and position diagnostic ablations

Seven predictors were compared across the same five random-variant and five position-held-out splits. Ridge alpha was selected on validation data and supervised models were refitted on training plus validation rows. In addition to overall Spearman correlation, the experiment reports:

- **Between-position Spearman:** correlation between the mean observed and predicted score at each test position.
- **Within-position centered Spearman:** pooled correlation after subtracting each test position's mean from observations and predictions.
- **Within-position pairwise accuracy:** fraction of unequal-score mutation pairs at the same position whose order is predicted correctly; prediction ties receive half credit, so `0.5` is chance/tied performance.
- **Macro within-position Spearman:** mean Spearman across positions with at least three test mutations and nonconstant observations and predictions.

Mathematically undefined correlations from constant predictions are recorded as missing rather than replaced with zero.

### Random-variant split: five-seed means

| Predictor | Overall Spearman | Between-position Spearman | Within-position centered Spearman | Pairwise accuracy | Macro within-position Spearman |
|---|---:|---:|---:|---:|---:|
| Training-position mean | 0.763634 | 0.823827 | — | 0.500000 | — |
| Mutation features | 0.405863 | 0.431490 | 0.326213 | 0.600940 | 0.217167 |
| Categorical position + mutation | 0.803717 | 0.860978 | 0.329136 | 0.603381 | 0.228494 |
| Zero-shot ESM-2 | 0.415531 | 0.379948 | 0.372621 | 0.631853 | 0.283691 |
| Frozen wild-type site only | 0.763745 | 0.824101 | — | 0.500000 | — |
| Frozen mutation difference only | 0.615305 | 0.709580 | 0.175273 | 0.548261 | 0.124892 |
| Frozen wild type + difference | **0.822962** | **0.878812** | **0.389255** | 0.623701 | **0.292794** |

The random-split headline is driven primarily by learning position sensitivity. A training-position mean already reaches `0.763634`, and the wild-type site representation alone reaches `0.763745` while having no ability to order substitutions within a position. Adding categorical position identity to the simple mutation baseline raises overall correlation from `0.405863` to `0.803717`, close to the combined frozen model's `0.822962`.

The combined frozen model still adds information: it exceeds wild-type-only by `0.059216` overall and wins that comparison on all five random splits. However, its within-position pairwise accuracy (`0.623701`) is slightly below zero-shot ESM-2 (`0.631853`). Its large overall advantage over zero-shot therefore does not primarily come from better ordering of alternative amino acids at the same position.

### Position-held-out split: five-seed means

| Predictor | Overall Spearman | Between-position Spearman | Within-position centered Spearman | Pairwise accuracy | Macro within-position Spearman |
|---|---:|---:|---:|---:|---:|
| Training-position mean | — | — | — | 0.500000 | — |
| Mutation features | 0.297238 | 0.188650 | 0.388313 | 0.622987 | 0.336005 |
| Categorical position + mutation | 0.291642 | 0.176880 | 0.388220 | 0.622942 | 0.336289 |
| Zero-shot ESM-2 | 0.437823 | 0.450667 | 0.371293 | 0.617544 | 0.321228 |
| Frozen wild-type site only | 0.378397 | 0.479248 | — | 0.500000 | — |
| Frozen mutation difference only | 0.348292 | 0.418120 | 0.275391 | 0.575169 | 0.204076 |
| Frozen wild type + difference | **0.446455** | **0.479521** | **0.404616** | **0.624471** | **0.338259** |

When test positions are completely unseen, the training-position mean falls back to one global constant and its correlation is undefined. Categorical position identity also loses its random-split advantage and performs like the original mutation baseline. This directly confirms that those controls depend on labeled measurements from the same positions.

The wild-type site embedding generalizes some position sensitivity to unseen residues: its between-position correlation is `0.479248`, but it cannot rank substitutions within a position. Mutation features show the opposite pattern: their between-position correlation is only `0.188650`, while their within-position pairwise accuracy (`0.622987`) is nearly identical to the combined ESM-2 model (`0.624471`). Combining wild-type context and mutation difference provides both types of signal.

Compared with zero-shot ESM-2, the combined frozen model improves mean overall correlation by only `0.008632` and wins three of five splits. Its mean improvements are `0.028855` for between-position Spearman, `0.033324` for position-centered Spearman, and `0.006928` for within-position pairwise accuracy. The supervised linear probe therefore adds modest complementary information, but the diagnostic does not establish a consistent large advantage over zero-shot scoring at unseen positions.

### Diagnostic conclusion

The `0.81` random-split result is mostly a measurement of how well the supervised model recovers position-level sensitivity after seeing other labeled mutations at those positions. It is not evidence of equally strong generalization to new residue sites. ESM-2's wild-type representation captures position tolerance, while mutation-specific ranking comes from a combination of simple mutation identity, zero-shot probabilities, and the mutation-difference representation.

Generated diagnostic files:

- `results/diagnostic_ablations/per_run.csv`
- `results/diagnostic_ablations/predictions.csv`
- `results/diagnostic_ablations/summary.csv`
- `results/diagnostic_ablations/summary.json`

## Next verification steps

The controlled LoRA and ranking experiment has been designed but has not yet been run. Its frozen
protocol, hypotheses, data-use rules, success criteria, and gated implementation plan are in
[`docs/LORA_EXPERIMENT_DESIGN.md`](docs/LORA_EXPERIMENT_DESIGN.md). The core masked-residual model,
fair-esm-compatible LoRA layers, and MSE/ListMLE objectives are implemented and covered by the
repository test suite. An integration check on the actual `esm2_t6_8M_UR50D` checkpoint replaced
all 36 planned linear modules, counted 552,960 adapter parameters, and preserved the initial model
output exactly. The validation-gated training interface now saves configuration, latest and best
checkpoints, history, predictions, and metrics; it also rejects configuration-mismatched resumes.
Synthetic frozen-MSE and LoRA-ListMLE smoke runs exercise this path without using the TEM-1 assay.
No TEM-1 pilot or LoRA result exists yet, and no LoRA metrics should be added to this log until the
corresponding commands complete successfully.

Next steps:

1. Review the training interface before the four-configuration technical pilot.
2. Run the validation-only five-epoch pilot after approval; do not evaluate test positions.
3. Preserve zero-shot ESM-2 as a required paired reference.
4. Reserve an additional verified assay for confirmation after model-selection decisions are frozen.
