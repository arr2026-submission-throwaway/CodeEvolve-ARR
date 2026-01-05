# CodeEvolve: Experiments and source code for the ARR submission
## Usage
To setup the proper conda environment, run the following:
```bash
conda env create -f environment.yml
conda activate codeevolve
```
The command-line version of codeevolve is implemented in ```src/codeevolve/cli.py```, and ```scripts/run.sh``` contains a bash script for running codeevolve on a given benchmark. The most important variables to be defined in this file are the ```API_KEY, API_BASE``` environment variables for connecting with an LLM provider.

## Experiments
All experimental results can be found in the ```experiments``` folder. All plots and data used for tables in the submission can be reproduced using the ```notebooks/experiment_analysis.ipynb``` notebook (use the above conda environment as kernel). The inputs for all benchmarks can be found in the ```problems``` folder.

