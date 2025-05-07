# FVOD: Artifact for FMCAD 2025

This repository contains the artifact accompanying our FMCAD 2025 paper:

**Title:** Towards Formal Verification of Deep Neural Networks for Object Detection.

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

---

Let me know if you need further assistance or additional modifications.


## 📄 License

MIT License

Permission is hereby granted to the FMCAD 2025 Program Committee to download, use, and execute this artifact solely for the purpose of artifact evaluation.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement.
