.PHONY: setup validate splits test zero-shot mutation-baseline esm-features esm-baseline repeated-position sample-efficiency repeated-sample-efficiency plot-sample-efficiency diagnostic-ablations

setup:
	python3 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -e .

validate:
	python scripts/validate_dataset.py

splits:
	python scripts/make_splits.py

test:
	python -m unittest discover -s tests -v

zero-shot:
	python scripts/score_zero_shot.py --device auto
	python scripts/evaluate.py results/esm2_masked_marginal.csv --output results/esm2_masked_marginal_all.json
	python scripts/evaluate.py results/esm2_masked_marginal.csv --split-strategy position --subset test --output results/esm2_masked_marginal_position_test.json
	python scripts/evaluate.py results/esm2_masked_marginal.csv --split-strategy random --subset test --output results/esm2_masked_marginal_random_test.json

mutation-baseline:
	python scripts/train_mutation_baseline.py --split-strategy position

esm-features:
	python scripts/extract_esm2_features.py --device auto

esm-baseline:
	python scripts/train_esm2_baseline.py --split-strategy position

repeated-position:
	python scripts/run_repeated_position_splits.py --seeds 42 43 44 45 46

sample-efficiency:
	python scripts/run_sample_efficiency.py --seeds 42 43 44 45 46 --fractions 0.01 0.05 0.10 0.25 0.50 1.0

repeated-sample-efficiency:
	python scripts/run_repeated_sample_efficiency.py --split-seeds 42 43 44 45 46 --sampling-seeds 1 2 3 4 5 --fractions 0.01 0.05 0.10 0.25 0.50 1.0

plot-sample-efficiency:
	python scripts/plot_sample_efficiency.py

diagnostic-ablations:
	python scripts/run_diagnostic_ablations.py --seeds 42 43 44 45 46
