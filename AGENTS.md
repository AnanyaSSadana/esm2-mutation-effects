# Codex project guidance

- Read `README.md` and `DATASET_CARD.md` before changing data or evaluation logic.
- Treat `data/raw/BLAT_ECOLX_Stiffler_2015.csv` and `data/provenance/DMS_substitutions.csv` as immutable source artifacts.
- Never fabricate metrics. Generated predictions and metrics belong under `results/` and are gitignored.
- Keep mutation positions one-based and relative to the full 286-residue target sequence. Do not subtract the 23-residue signal peptide.
- For supervised comparisons, use the same split strategy and report it with every metric. Prefer the `position` split for the primary result.
- Use validation data for hyperparameter selection and evaluate the test subset only after the choice is fixed.
- Run `python scripts/validate_dataset.py` and `python -m unittest discover -s tests -v` after data or split changes.
