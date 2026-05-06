# CodeEvolve Experiments

Experimental results for CodeEvolve ARR Submission.

## Overview

This repository provides:

- **Experimental configurations** for reproducing all results
- **Raw experimental data** from paper runs (`.pkl`, `.py`, `.txt` files)
- **Analysis notebooks** with visualizations and statistical tests

## Repository Structure

```
science-codeevolve-experiments/
├── experiments/          # Raw experimental results
├── notebooks/           # Analysis and visualization
│   ├── experiment_analysis.ipynb       # Main analysis notebook
│   └── figs/                           # Generated figures from paper
└── README.md
```

### Directory Details

- **`experiments/`**: Contains results from paper experiments including:
  - Solution histories (`.py` files)
  - Checkpoints (`.pkl` files)
  - Logs and metadata (`.txt` files)
  - Multiple runs with different seeds/configurations

- **`notebooks/`**: Jupyter notebooks for analysis
  - `experiment_analysis.ipynb`: Statistical analysis and comparisons

## Prerequisites

### Configure LLM API Access

Set your LLM API credentials as environment variables:

```bash
export API_KEY=your_api_key_here
export API_BASE=your_api_base_url
```

## Reproducibility

This repository supports two distinct notions of reproducibility:

#### 1) Reproducing the paper analysis (deterministic, using included artifacts)
The folder `experiments/` contains the raw artifacts used in the paper (checkpoints, histories, logs). The notebook(s) in `notebooks/` analyze those artifacts to generate the plots and comparisons. Re-running the analysis should reproduce the reported figures/tables as long as your analysis environment is compatible.

#### 2) Re-running the full search (best-effort; exact replay depends on the LLM provider)
**Exact numerical reproduction of a full evolutionary run is not guaranteed** when using hosted LLM APIs.

Why:
- Many commercial LLM providers **do not support deterministic sampling** or **do not honor `seed`**.
- Even when a provider accepts `seed`, outputs can vary due to backend nondeterminism (load balancing, infrastructure-level randomness, model version rollouts).

This is not a limitation of CodeEvolve’s evolutionary framework: CodeEvolve is **seedable for its internal stochastic decisions**, and it forwards model `seed` to OpenAI-compatible endpoints when supported. The remaining nondeterminism comes from the LLM backbone/provider.
