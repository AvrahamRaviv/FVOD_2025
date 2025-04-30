# FMCAD 2025 Artifact: Towards Formal Verification of Deep Neural Networks for Object Detection

This repository contains the artifact accompanying our FMCAD 2025 paper:

**Title:** Towards Formal Verification of Deep Neural Networks for Object Detection  
**Authors:** [Redacted for double-blind review]

## Repository Structure

```
├── train/      # Pre-processing scripts and data preparation  
├── eval/       # Modified abcrown with core IoU verification algorithm  
├── analyze/    # Post-processing and result analysis
├── data/       
```

- **train/** and **analyze/**: Contain scripts for pre- and post-processing related to our algorithm.  
- **eval/**: Hosts the main implementation of our Intersection over Union (IoU) verification algorithm.

## Requirements

This artifact builds upon the [alpha-beta-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN) framework.  
Please follow their installation instructions.  
Our repository uses only modified pure-Python scripts and introduces no additional dependencies.

## License

MIT License

Permission is hereby granted to the FMCAD 2025 Program Committee to download, use, and execute this artifact solely for the purpose of artifact evaluation.

The software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement.