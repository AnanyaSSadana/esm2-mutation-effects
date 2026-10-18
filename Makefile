.PHONY: setup validate splits test zero-shot mutation-baseline esm-features esm-baseline repeated-position

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
