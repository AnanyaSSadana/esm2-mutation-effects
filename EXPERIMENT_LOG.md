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

## Validation-only LoRA technical pilot

The four controlled masked-residual configurations completed a five-epoch technical pilot on
position split 42 with training seed 1. This pilot was designed to verify optimization, gradient
flow, runtime, memory, checkpoint selection, and MPS compatibility. It is not a test result:
test positions were not evaluated, and no test prediction or metric files were created.

All configurations began at the same epoch-zero zero-shot validation Spearman correlation of
`0.413935`. The table reports the best validation checkpoint selected during the five epochs.

| Configuration | Best epoch | Validation Spearman | Change from epoch zero | Between-position Spearman | Within-position centered Spearman | Pairwise accuracy | Total epoch time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen + MSE | 1 | 0.418450 | +0.004515 | 0.385983 | 0.433707 | 0.630004 | 29 s |
| Frozen + ListMLE | 3 | **0.500050** | **+0.086115** | 0.534359 | 0.417628 | 0.620558 | 503 s |
| LoRA + MSE | 1 | 0.420914 | +0.006979 | 0.385299 | 0.432946 | 0.632254 | 72 s |
| LoRA + ListMLE | 1 | 0.496686 | +0.082751 | **0.578120** | **0.443015** | 0.625281 | 1,653 s |

ListMLE produced the material validation improvement in this pilot. Its frozen-backbone result
improved overall validation Spearman by about `0.086` over the shared zero-shot initialization;
LoRA plus ListMLE improved it by about `0.083`. In contrast, both MSE variants improved overall
Spearman by less than `0.007` and selected epoch 1 even though their regression losses continued
to fall. This divergence reinforces the importance of selecting checkpoints using validation
Spearman rather than training or validation MSE.

This single split and optimization seed does not establish that LoRA helps or fails. LoRA plus
MSE was only `0.002464` above Frozen plus MSE, while LoRA plus ListMLE was `0.003364` below Frozen
plus ListMLE in overall Spearman. However, the LoRA ListMLE checkpoint had stronger between-position
and position-centered correlations than Frozen plus ListMLE, so the models may be changing
different aspects of the ranking.

LoRA adapters received nonzero gradients in both objectives. Maximum observed adapter gradient
norms were `4.570` for MSE and `26.581` for ListMLE, confirming that adaptation was active. Global
pre-clipping gradient norms frequently exceeded the configured limit of `3.0`, and clipping was
therefore operational. No non-finite losses or predictions were observed.

Runtime is the main feasibility issue before the multi-seed development phase. Frozen plus
ListMLE required about 8.4 minutes for five epochs, and LoRA plus ListMLE required about 27.6
minutes. The current implementation processes global ranking lists in small microbatches, causing
many repeated ESM-2 forward passes. Before scaling the experiment, the frozen path should cache its
unchanging masked-site representations, and larger LoRA list microbatches should be benchmarked
without changing the loss, effective batch, split, or test policy.

The frozen representation cache has now been implemented. It computes the 263 masked-site
representations once and trains only the residual head against that fixed tensor. On the actual
ESM-2 checkpoint and TEM-1 inputs, cached and direct predictions were exactly equal for an
80-variant check (`maximum_absolute_difference = 0.0`). This is therefore an implementation
optimization, not a protocol amendment. A validation-only benchmark runner is ready to compare
LoRA ListMLE microbatch sizes 8 and 16 while preserving the effective batch of 16 lists.

### LoRA ListMLE batching benchmark

The validation-only batching benchmark completed on MPS using position split 42 and training
seed 1. Both runs used one epoch, 16 effective lists per optimizer step, and the same model,
objective, data ordering, and optimization settings. Neither run evaluated test positions.

| List microbatch size | Epoch time | Training loss | Validation Spearman | Status |
|---:|---:|---:|---:|---|
| 8 | 793.57 s | 5.711611 | 0.496751 | Completed |
| 16 | 1,407.77 s | 5.711689 | 0.496751 | Completed |

Microbatch 8 reduced epoch time by 43.6% (a 1.77x speedup relative to microbatch 16). The two
training losses differed by only `0.000078`, and validation Spearman differed by only
`0.0000006`, so the benchmark provides no evidence that the computational partition changed the
optimization result. The reported MPS allocation at epoch end is not treated as a peak-memory
measurement.

Microbatch 8 was initially selected from these two candidates based on runtime and successful
completion, not validation quality. That conclusion was later superseded when the original
microbatch-4 pilot timing was included in the comparison.

The five saved LoRA+ListMLE pilot epochs at microbatch 4 required 279.89–357.79 seconds each on
MPS, materially less than either later candidate. A direct CPU comparison likewise favored size 4
(`781.56` seconds) over size 8 (`805.70` seconds). Microbatch 4 is therefore fixed for subsequent
development and confirmation runs. The effective batch remains 16 lists, and no test positions
were evaluated in any throughput check.

Before restarting Phase 2 after the lossless-ranking-target correction, the batching benchmark was
upgraded to make a direct post-correction comparison. It ran microbatch sizes 4, 8, and 16 for one
epoch each using the identical split-42, sampling-seed-1, 2,004-label manifest, ordinal ranks
derived from float64 assay scores, and an effective batch of 16 lists. No test positions were
evaluated.

| List microbatch size | Epoch time | Training loss | Validation Spearman |
|---:|---:|---:|---:|
| 4 | 129.57 s | 6.419556 | 0.477690 |
| 8 | 163.63 s | 6.419550 | 0.477686 |
| 16 | 4,104.25 s | 6.419528 | 0.477717 |

Microbatch 4 was 20.8% faster than size 8 and 31.7 times faster than size 16 on MPS. The maximum
validation-Spearman difference among candidates was only `0.000031`, and the training losses and
gradient diagnostics were likewise nearly identical. The direct controlled benchmark therefore
confirms microbatch 4 as the local MPS setting. These are technical throughput measurements, not
scientific results. The full machine-readable record is retained at
`results/phase2_microbatch_benchmark/benchmark_summary.json`.

The pilot validation trajectories also supported a feasibility amendment from early-stopping
patience 15 to 5. The selected epochs were 1 for Frozen+MSE, LoRA+MSE, and LoRA+ListMLE and 3 for
Frozen+ListMLE; every later observed validation correlation declined through epoch 5. Patience 5
still permits five consecutive non-improving epochs before stopping while avoiding ten additional
expensive epochs for every LoRA+ListMLE run. This amendment was made before development training
and before any confirmatory test evaluation.

The Milestone 5 runner is now implemented as `scripts/run_lora_development.py`. It fixes position
split 42, optimization seeds 1–3, all four controlled configurations, maximum 100 epochs,
early-stopping patience 5, 16 effective ListMLE lists per optimizer step, and list microbatch 4.
The runner exposes no test-evaluation option, rejects test artifacts in development directories,
checks saved configurations against the fixed protocol, resumes interrupted runs, skips completed
runs, and updates a cumulative validation-only summary after every completed run. Its 12-command
development matrix later completed validation-only; the protocol was locked before confirmation.

Generated pilot files:

- `results/lora_pilot/pilot_summary.json`
- `results/lora_pilot/<configuration>/split_42/seed_1/`

## Phase 1: protocol-locked controlled LoRA experiment

Phase 1 is complete. The development stage ran all four configurations on position split 42 and
optimization seeds 1–3 without evaluating test labels. Its resulting protocol lock has SHA-256
`c7ad1f57448e136cbe19fae46d996370f520403eccfef591073640d6b3d4001b`. The confirmation stage then
completed all `4 × 5 × 3 = 60` runs: five position-held-out splits (42–46), three optimization
seeds per configuration, one validation-selected checkpoint, and one test evaluation per saved
checkpoint. The first split reused the locked development checkpoints; splits 43–46 were newly
trained under the identical lock.

### Primary result: held-out-position Spearman

Each table entry below is a split-level mean across three optimization seeds. The aggregate
standard deviation is therefore across the five independent held-out-position splits—not across
the 15 optimization runs.

| Configuration | Split 42 | Split 43 | Split 44 | Split 45 | Split 46 | Mean ± split SD |
|---|---:|---:|---:|---:|---:|---:|
| Frozen + MSE | 0.436275 | 0.545161 | 0.319425 | 0.471889 | 0.188505 | 0.392251 ± 0.140070 |
| Frozen + ListMLE | 0.498554 | 0.631315 | 0.362272 | 0.394024 | 0.292402 | 0.435714 ± 0.132173 |
| LoRA + MSE | 0.438464 | 0.622042 | 0.325775 | 0.471889 | 0.207186 | 0.413071 ± 0.156351 |
| LoRA + ListMLE | **0.496694** | **0.649853** | **0.372251** | **0.471889** | **0.374791** | **0.473096 ± 0.113617** |
| Zero-shot ESM-2 | 0.426226 | 0.656576 | 0.342814 | 0.471889 | 0.291610 | 0.437823 ± 0.141026 |

ListMLE improves the controlled frozen model by `+0.043462` over Frozen + MSE and improves the
LoRA model by `+0.060025` over LoRA + MSE. LoRA + ListMLE has the best aggregate mean and the
lowest split-level standard deviation among the four controlled configurations.

### Predeclared decision criteria

| Criterion for LoRA + ListMLE | Result | Decision |
|---|---:|---|
| Beat zero-shot by at least 0.02 mean Spearman | `+0.035273` | Met |
| Beat Frozen + ListMLE under the same protocol | `+0.037382` | Met |
| Beat zero-shot on at least four of five splits | 3 wins, 1 tie, 1 loss | Not met |
| Avoid material within-position ranking degradation | centered rho: 0.402643 vs 0.412772; pairwise accuracy: 0.626976 vs 0.630829 | Comparable, small decrease |
| Avoid dependence on one optimization seed | per-split seed SD: 0.000000–0.034826 | Met |

The outcome is therefore **qualified positive evidence**: LoRA + ListMLE improves mean unseen-
position performance and is stable across optimization seeds, but its zero-shot advantage is not
uniform across position splits. The strict predeclared success rule is not fully met.

The pattern also separates the two interventions. The ListMLE objective produces the larger gain,
while LoRA adds little under MSE and provides its clear advantage when combined with ListMLE.
LoRA + ListMLE raises mean between-position Spearman to `0.493835` from Frozen + ListMLE's
`0.425322`, while retaining comparable within-position metrics. This is a TEM-1-specific result;
it needs a separate assay for external confirmation.

Commands completed:

```bash
python scripts/run_lora_development.py --device auto
python scripts/lock_lora_protocol.py
python scripts/run_lora_confirmatory.py --device auto
# Equivalent resumable orchestration command:
python scripts/run_phase1.py --device auto
```

Generated Phase 1 artifacts (gitignored):

- `results/lora_development/development_summary.json`
- `results/lora_development/protocol_lock.json`
- `results/lora_confirmatory/confirmatory_summary.json`
- `results/phase1_execution_summary.json`

## Phase 2: protocol-locked label-efficiency experiment

Phase 2 is complete. Its first incomplete matrix was invalidated after a float32 rounding collision
merged two distinct ListMLE targets; those artifacts remain under
`results/phase2_invalidated_float32_target_collision_2026-10-20/` and are excluded. The clean run
used lossless ordinal ranking targets, cross-run frozen-representation caching, and compact
trainable-only checkpoints. A validation-only NVIDIA L4 benchmark selected ListMLE microbatch 16
before the new protocol was locked.

The completed protocol hash is
`ecd213c9645890ca90656059867916c78e3a879cec72e9ebf1b46ca2d60cddad`. Validation of the
downloaded summary found all 190 expected supervised run identities exactly once: 95 Frozen +
ListMLE and 95 LoRA + ListMLE runs, balanced across five position splits, all label budgets, and
the required subset samples. No primary metric was missing or non-finite. The analyzed summary
file has SHA-256 `35d817b5095db84e4bfb301428dac18997559439f9e2308f48cf4c84707c5d92`.

### Primary label-efficiency result

Each value below is the mean and sample standard deviation across the five held-out-position split
means. Three nested subset samples are averaged within each split below the 4,008-label budget.

| Labels | Frozen + ListMLE | LoRA + ListMLE | LoRA − zero-shot | LoRA − frozen |
|---:|---:|---:|---:|---:|
| 20 | 0.426817 ± 0.128307 | 0.425522 ± 0.128105 | -0.012300 | -0.001295 |
| 40 | 0.427675 ± 0.132888 | 0.426235 ± 0.134590 | -0.011586 | -0.001440 |
| 200 | 0.441302 ± 0.131955 | 0.442547 ± 0.133761 | +0.004726 | +0.001246 |
| 401 | 0.436298 ± 0.124218 | 0.442377 ± 0.127471 | +0.004555 | +0.006079 |
| 1,002 | 0.440744 ± 0.133436 | 0.450552 ± 0.127444 | +0.012730 | +0.009807 |
| 2,004 | 0.440929 ± 0.123726 | **0.466493 ± 0.117322** | **+0.028671** | **+0.025564** |
| 4,008 | 0.436906 ± 0.130708 | 0.461451 ± 0.119521 | +0.023630 | +0.024546 |

Zero-shot ESM-2 is `0.437822 ± 0.141027`. LoRA underperforms zero-shot at 20 and 40 labels,
becomes approximately competitive at 200–401, and improves progressively at larger budgets. The
pre-specified practical mean advantage of `+0.02` is first met at 2,004 labels. Performance is not
strictly monotonic: the 2,004-label mean is slightly higher than the 4,008-label mean.

At 2,004 labels, the paired LoRA-minus-zero-shot differences across the five splits range from
`-0.007343` to `+0.076464`, with three wins, one tie, and one loss. This is **qualified positive
evidence** that sufficient TEM-1 labels improve mean unseen-position performance; it does not
establish uniform improvement across possible held-out residue sets.

### What LoRA learns and where uncertainty comes from

At 2,004 labels, LoRA improves between-position Spearman over Frozen + ListMLE (`0.482758` versus
`0.456228`) while slightly reducing position-centered Spearman (`0.389073` versus `0.399127`) and
pairwise accuracy (`0.621158` versus `0.626226`). The LoRA gain therefore comes mainly from better
ranking the sensitivity of unseen positions, consistent with the Phase 1 diagnostic.

The split-level standard deviation for LoRA at 2,004 labels is `0.117322`, whereas the mean
within-split subset-sampling standard deviation is only `0.014464`. Biological partition choice—
which positions are held out—dominates uncertainty from the particular labeled mutations sampled
within the training positions.

Generated analysis artifacts (gitignored):

- `results/phase2_label_efficiency/analysis/phase2_learning_curves.png`
- `results/phase2_label_efficiency/analysis/paired_comparisons.csv`
- `results/phase2_label_efficiency/analysis/metric_summary.csv`
- `results/phase2_label_efficiency/analysis/analysis_summary.json`

## Next steps

Phase 2 supports moving to the planned multi-assay/meta-learning phase. The concrete research
question is now whether auxiliary DMS assays can shift the LoRA learning curve left—achieving the
2,004-label benefit with substantially fewer labels—while preserving position-held-out and
assay-held-out evaluation. W&B can be added as a monitoring layer by backfilling the saved Phase 2
histories; repository JSON and CSV artifacts remain the authoritative scientific record.
