### CS 598 Reproduction of EHRMamba

Paper: https://arxiv.org/abs/2405.14567 (EHRMamba: Towards Generalizable and Scalable Foundation Models for Electronic Health Records)

Daniel Kwan (NetID: dwkwan2)

## Installation

(no `requirements.txt` provided due to manual installation steps needed)

1. Create a Conda environment with Python 3.12, PyTorch 2.9.1 and CUDA 13.0 (we had issues installing/compiling `mamba_ssm` on any lower CUDA version)

```
conda create -n ehrmamba python=3.12
conda activate ehrmamba
pip3 install torch=2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

2. Install at least PyHealth v2.0 (you may need to manually git clone the repository, remove version requirements in `pyproject.toml` such as requiring PyTorch 2.7.1, etc. Otherwise, it may force use a lower CUDA version i.e. 12.6 and therefore preventing installation of later packages)

```
git clone https://github.com/sunlabuiuc/PyHealth
# cd into directory
# Remove version requirements in pyproject.toml
pip install -e .
```

3. Install mamba.py library for their simple pure PyTorch implementation of Mamba blocks and architecture (also used in `transformers`, we bypass HF by installing this directly)

```
pip install mambapy
```

4. Install the `click` library (used for easy CLI building)

```
pip install click
```

4. (optional) Install `mamba_ssm` and `causal_conv1d`. Currently, there are open issues installing from pip, so we suggest cloning the repositories and using `pip install -e . --no-build-isolation` (to prevent errors about CUDA version mismatch, which occur even if you install)

## Overview

Please see `runner.py` and `main.py` for instructions on how to run the code. Changes related to PyHealth are self-contained in this repository.

- `runner.py` - will automatically run all experiments in presented in the report, calling `main.py` will the necessary arguments
- `main.py` - handles individual experiment runs, training on the specified clinical predictive task using the specified model and other arguments. Please see `--help` for a list of all options.
- `models/`
    - `models/mamba_mpy.py` - main implementation of EHRMamba-style architecture made compatible with PyHealth
    - `models/mamba2_mpy.py` - extension implementation replacing with Mamba2 blocks
    - `models/mamba_mpy_add.py` - (unused, provided for reference) follows EHRMamba closely in adding together embeddings before passing into the model; as mentioned in report, performs much worse and due to limited compute time, was not run fully in experiments
    - `models/jamba_mpy.py` (unused, for reference) alternative LLM suggestion of replacing Mamba blocks with Jamba-style (hybrid Mamba architecture using some attention layers); performs worse, likely due to some lingering bugs that we were unable to fully resolve - again, not run fully due to limited compute time.
    - `models/mamba.py` (unused, for reference) original LLM-assisted implementation using `mamba_ssm` instead of `mambapy`; has some bugs and we were not fully satisfied with the implementation
- `tasks/`
    - `tasks/tasks.py` - implements PyHealth-compatible tasks for mortality and LOS prediction to match exactly the descriptions as in the EHRMamba paper (see report for exact differences versus existing PyHealth tasks)
- `utils/`
    - `utils/optimization.py` - helper code to build LR scheduler with linear warmup and decay
    - `utils/trainer.py` - modified version of PyHealth's Trainer class, to add support for LR schedulers
- `datasets/mimic-iv-2.2` - expected path for MIMIC-IV dataset (can be changed in `main.py`)



