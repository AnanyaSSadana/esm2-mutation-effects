# Google Colab runner

This folder contains the upload-ready Phase 2 workflow.

Files:

- `01_cuda_benchmark.ipynb`: verifies CUDA, installs the project, creates the required zero-shot
  predictions, and benchmarks ListMLE microbatch sizes 4, 8, and 16 without evaluating test data.
- `02_run_phase2.ipynb`: reads the winning CUDA microbatch, validates the repository, and runs or
  resumes the complete protocol-locked 190-run Phase 2 matrix.
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

Do not change the chosen microbatch after the smoke run or protocol lock has been created. The
runner deliberately rejects configuration mismatches instead of mixing incompatible runs.
