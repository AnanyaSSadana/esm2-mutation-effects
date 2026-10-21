# Google Colab runner

This folder contains upload-ready Colab workflows.

Files:

- `01_cuda_benchmark.ipynb`: verifies CUDA, installs the project, creates the required zero-shot
  predictions, and benchmarks ListMLE microbatch sizes 4, 8, and 16 without evaluating test data.
- `02_run_phase2.ipynb`: reads the winning CUDA microbatch, validates the repository, and runs or
  resumes the complete protocol-locked 190-run Phase 2 matrix.
- `03_run_phase3a_pilot.ipynb`: mounts Google Drive, clones the public repository into a separate
  Drive folder, verifies the attached CUDA GPU, and runs the bounded Phase 3A pilot.
- `04_run_phase3b_lite.ipynb`: reuses the completed Phase 3A Drive checkout and runs the four
  newly locked target splits for the compact Phase 3B-lite replication.
- `esm2-mutation-effects-colab.zip`: the project bundle requested by both notebooks. This generated
  file is intentionally ignored by Git.

## Run order

1. In Colab, choose **Runtime > Change runtime type > GPU**. An L4 is a good fit when offered; the
   8M-parameter model does not need an A100.
2. Upload and open `01_cuda_benchmark.ipynb`, then run all cells. When prompted, upload
   `esm2-mutation-effects-colab.zip`. The notebook saves the bundle and benchmark under
   `MyDrive/esm2-phase2-colab/`.
3. Open `02_run_phase2.ipynb` and run all cells. It reuses the saved bundle and results, so there is
   normally no second upload.
4. If Colab disconnects, reconnect to a GPU and rerun notebook 02 from the top. Completed runs are
   skipped and an incomplete run resumes from `last.pt`.

Keep the notebook tab connected while training. Pay-as-you-go compute units do not by themselves
include guaranteed background execution, and Colab runtime availability and lifetime remain
dynamic.

## Phase 3A notebook

Before opening `03_run_phase3a_pilot.ipynb`, push the current Phase 3A commits to GitHub. In VS
Code, open the notebook, select the connected **Colab GPU** kernel, and run its cells from top to
bottom. It uses `MyDrive/esm2-phase3a/`, separate from the Phase 2 Drive folder, so an interruption
can be recovered before target-test predictions have been written. Once the pilot completes, do
not re-run the launch cell: the protocol deliberately opens the held-out target test positions only
once.

## Phase 3B-lite notebook

Run `04_run_phase3b_lite.ipynb` only after Phase 3A has completed in the same
`MyDrive/esm2-phase3a/` checkout. The notebook pulls the new code, verifies CUDA, locks all four
new target manifests before evaluation, and runs the resumable matrix. It never reruns a completed
split, but it deliberately refuses to reopen a split that has test predictions without a matching
summary. Once its matrix summary exists, do not rerun the launch cell.

Do not change the chosen microbatch after the smoke run or protocol lock has been created. The
runner deliberately rejects configuration mismatches instead of mixing incompatible runs.
