# Motivating Study

This repository contains the motivating study results of [PCREQ: Automated Inference of Compatible Requirements for Python Third-party Library Upgrades](https://arxiv.org/abs/2508.02023).

## pip_experiment directory

The `pip_experiment` directory contains the raw installation and execution outputs for each upgrade experiment. Specifically, it includes the logs of pip install commands (used to detect version compatibility issues) and the corresponding project execution results (used to identify runtime code compatibility issues). 

Note: The file `installationerror.txt` is only generated if the pip install command fails (i.e., when installation errors occur).

### **Example Directory Layout**

```bash
./
├── 3d-ken-burns
    ├── torch
        ├── 1.7.1
            ├── executionresult.txt        # corresponding project execution results (used to identify runtime code compatibility issues). 
            ├── installationerror.txt      # logs of pip install commands (used to detect version compatibility issues) 
            ├── requirements.txt           # the requirements.txt file
        ...
        └──
    ├── torchvision
        ├── 0.8.1
            ├── executionresult.txt        # corresponding project execution results (used to identify runtime code compatibility issues). 
            ├── requirements.txt           # the requirements.txt file
        ...
        └──
    ...
    └──
```

## experiment-result.xlsx

The `experiment-result.xlsx` file summarizes all 2,095 upgrade experiments from the PCREQ paper's motivating study. It includes detailed information such as the target library, version range, pip installation outcome, runtime execution result, compatibility issue type (VCI or CCI), failure patterns (Pattern a–e), and fine-grained CCI categories (module, API name, parameter, and body). This file serves as the primary data source for the statistical analysis and findings presented in Section 2 of our paper.

## run_proj.py

The script used to run the project after a dependency upgrade, to detect CCIs through actual execution.

## Project source code and related virtual environment
The project source code is available on [Zenodo](https://doi.org/10.5281/zenodo.16100741), and the corresponding virtual environment can be found at link.md.

### **Example Directory Layout**

```bash
./
├── 3d-ken-burns-torch-1.7.1.tar.gz        # Conda environment
```

### :hammer_and_wrench: **Extracting and Using Conda Environments**

To extract and use a downloaded Conda environment, run:
```bash
cd /home/usr/anaconda3/envs
mkdir envName
tar -xzvf envName.tar.gz -C envName
```
