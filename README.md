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

## 📦 Requirements

This artifact builds upon the [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) framework.  
Please follow their installation instructions.  
Our repository uses only modified pure-Python scripts and introduces no additional dependencies.

## 📄 License

MIT License

Permission is hereby granted to the FMCAD 2025 Program Committee to download, use, and execute this artifact solely for the purpose of artifact evaluation.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement.
