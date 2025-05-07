# FVOD: Artifact for FMCAD 2025

This repository contains the artifact accompanying our FMCAD 2025 paper:

A static snapshot of the artifact is available at: [https://zenodo.org/records/15355735](https://zenodo.org/records/15355735)  
DOI: [10.5281/zenodo.15355735](https://doi.org/10.5281/zenodo.15355735)  
SHA256 checksum: `278d80012405bfaf24bd7443f09e88ed61dd254c43cbceae752bb3c8dae1604f`


**Towards Formal Verification of Deep Neural Networks for Object Detection.**

## 📁 Repository Structure

```
 ├── abcrown/    # Extended version of alpha-beta-CROWN with OD/IoU verification functionality  
 ├── train/      # Pre-processing scripts (training models, etc.) and data preparation  
 ├── analyze/    # Post-processing and result analysis  
 ```

- **abcrown/**: Forked and extended version of [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN), adding support for object detection and IoU verification.
- **train/** and **analyze/**: Contain pre- and post-processing utilities.
- The main entry point is [`abcrown/FVOD/verify_fc.py`](abcrown/FVOD/verify_fc.py), which demonstrates minimal example usage. You can customize arguments as needed.

## 📦 Requirements and Installation

The artifact was tested on a Mac with an Apple M3 chip (8-core CPU, 8-core GPU), 16GB RAM, running macOS 15.4.1 (24E263).

This artifact builds upon the [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) framework. Our repository is based on the April 2024 release of alpha-beta-CROWN (commit `1a3533a`).

To set up the environment:

1. **Clone the repository and checkout the specific commit:**

   ```bash
   git clone --recursive https://github.com/Verified-Intelligence/alpha-beta-CROWN.git
   cd alpha-beta-CROWN
   git checkout 1a3533a
   ```

2. **Create and activate the conda environment:**

   ```bash
   conda create -n alpha-beta-crown python=3.11 -y
   conda activate alpha-beta-crown
   ```

3. **Install dependencies:**

   ```bash
   pip install -r complete_verifier/requirements.txt
   ```
Our repository uses only modified pure-Python scripts and introduces no additional dependencies beyond those required by alpha-beta-CROWN.

After installing alpha-beta-CROWN, replace its source code with our modified version provided in the `abcrown/` folder of this repository.

Here’s the `Usage` section to add to your `README.md`:

## Usage

### Minimal Example

After completing installation and replacing the alpha-beta-CROWN source with our modified `abcrown/` folder, you can run a minimal verification example using:

```bash
python abcrown/FVOD/verify_fc.py --config abcrown/complete_verifier/exp_configs/OD/d_loc_init.yaml
````

This uses a configuration file for fast evaluation and correctness check.

### General Usage

To run verification with any custom configuration YAML:

```bash
python abcrown/FVOD/verify_fc.py --config path/to/your_config.yaml
```

You may customize model path, dataset, perturbation bounds, and verification settings directly in the YAML file.

## 📄 License

MIT License

Permission is hereby granted to the FMCAD 2025 Program Committee to download, use, and execute this artifact solely for the purpose of artifact evaluation.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement.
