# Dataset card: BLAT_ECOLX_Stiffler_2015

## Selection rationale

This repository uses the ProteinGym processed single-substitution assay `BLAT_ECOLX_Stiffler_2015` (TEM-1 beta-lactamase, *Escherichia coli*).

It was selected because:

- the original experiment is well documented and measured essentially all single amino-acid substitutions in the mature enzyme;
- ProteinGym contains 4,996 unique single-mutant rows and zero multiple mutants;
- 263 consecutive positions are covered (24–286), usually with all 19 alternatives;
- the 286-residue reference sequence is far below the 1,022-residue limit of `esm2_t6_8M_UR50D`;
- the biological readout—growth under ampicillin selection—is clear and relevant to protein function.

## Experimental context

Stiffler, Hekstra, and Ranganathan measured TEM-1 beta-lactamase variants under a range of ampicillin concentrations. ProteinGym's processed assay uses the raw phenotype column named `2500`, corresponding to the stringent 2,500 µg/mL ampicillin condition. The source paper reports 4,997 amino-acid mutations in the experiment; the ProteinGym processed table contains 4,996 scored rows after its processing/filtering.

This is an organismal-fitness proxy, not a direct thermodynamic stability measurement. Scores can reflect expression, folding, catalytic activity, and the nonlinear link from resistance activity to cell growth.

## Fields

- `mutant`: one-letter wild type, one-based full-sequence position, one-letter alternate (for example `H24C`).
- `mutated_sequence`: full 286-residue variant sequence.
- `DMS_score`: ProteinGym-direction-normalized experimental score; higher is better.
- `DMS_score_bin`: ProteinGym binary label using the assay median cutoff of `-1.159498916`.

The binary label is retained for provenance but is not used by the regression baselines.

## Integrity checks performed

- 4,996 rows and 4,996 unique mutation IDs.
- No missing values.
- All sequences are length 286.
- Every mutation's wild-type residue agrees with the authoritative target sequence.
- Every `mutated_sequence` differs from the target at exactly the declared site and contains the declared alternate residue.
- Mutated positions span 24–286 and total 263 unique positions.
- `DMS_score` ranges from `-3.743276064` to `0.355790663`.
- Mutant IDs, scores, and binary labels exactly match ProteinGym's earlier official Hugging Face mirror.

Run `python scripts/validate_dataset.py` to repeat the row-level checks.

## Splits

`data/splits/splits.csv` stores two deterministic seed-42 strategies:

- `random`: 80% train, 10% validation, 10% test at the mutation-row level.
- `position`: 80%/10%/10% after shuffling the 263 mutated residue positions; all substitutions at a position stay together.

The position-held-out split is the default for supervised scripts because it avoids same-position leakage. These are project splits, not claims of reproducing ProteinGym's official five-fold supervised benchmark splits.

## Provenance and license

The assay was extracted from `DMS_ProteinGym_substitutions.zip` in the authoritative ProteinGym v1.3 Zenodo record, DOI `10.5281/zenodo.15293562`. The repository also retains the record's authoritative `DMS_substitutions.csv` reference file.

The Zenodo API metadata for record 15293562 reports license identifier `mit-license`. The ProteinGym GitHub repository is also MIT-licensed. The original Cell article has its own publisher terms; cite it when using this assay. This notice records provenance and is not legal advice.

## Limitations

- One protein and one selection condition cannot establish broad protein-family generalization.
- The score distribution is strongly shaped by a stringent growth selection and may be nearly threshold-like.
- Random row splits can overestimate generalization because the same residue position appears across subsets.
- ESM-2 pretraining data may include homologous sequences; this project does not claim a fully decontaminated benchmark.
- Correlation measures ranking agreement, not calibrated biological effect size or causal mechanism.
