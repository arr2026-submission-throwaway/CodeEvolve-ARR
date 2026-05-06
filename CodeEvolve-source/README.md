# CodeEvolve ARR Submission

Source code for CodeEvolve ARR Submission.

## Quick Start

### Installation

Clone this repository and create the conda environment:

```bash
conda env create -f environment.yml
conda activate codeevolve
```

### Basic Usage

Configure your LLM provider by setting environment variables:

```bash
export API_KEY=your_api_key_here
export API_BASE=your_api_base_url
```

> `API_BASE` must point to an **OpenAI-compatible** API base URL (hosted provider, gateway, or local inference server).

Run CodeEvolve via the command line:

```bash
codeevolve \
  --inpt_dir=INPT_DIR \
  --cfg_path=CFG_PATH \
  --out_dir=RESULTS_DIR \
  --load_ckpt=LOAD_CKPT \
```

**Arguments:**
- `--inpt_dir`: Directory containing the evaluation script and the initial codebase
- `--cfg_path`: Path to YAML configuration file (required for new runs)
- `--out_dir`: Directory where results will be saved
- `--load_ckpt`: Checkpoint to load (0 for new run, -1 for latest, or specific epoch)

The `scripts/run.sh` provides a bash script for running CodeEvolve with `taskset` to limit CPU usage. See `src/codeevolve/cli.py` for further details.
