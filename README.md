# EpiAbSpec: Antibody–Epitope Prediction

This repository contains an antibody–epitope binding prediction pipeline based on protein structural features and a Graph Transformer model.

Given an antibody identifier with its FASTA sequence and V/J annotation, the pipeline predicts a score for every epitope in `Epitope_lib` and writes a ranked CSV file.

## Pipeline

```text
Antibody FASTA + V/J annotation
        |
        v
antibody structure (IgFold + PyRosetta)
        |
        v
DSSP + PSSM + AbLang + backbone coordinates
        |
        v
antibody graph paired with every epitope graph
        |
        v
five GraphTrans checkpoints, averaged
        |
        v
output/<antibody>_pred.csv
```

> **Note:** The current model uses the antibody heavy chain for graph construction and V/J features. The light-chain sequence is accepted by the FASTA reader and passed to IgFold, but light-chain features are not used by the trained predictor.

## Repository Layout

All files and resource directories are located directly under the repository root.

```text
EpiAbSpec/
├── antibody/                # generated/intermediate antibody data
├── Epitope_lib/             # epitope structures, features, and metadata
├── oas_fasta/               # antibody FASTA inputs
├── oas_VJ/                  # antibody V/J annotations
├── output/                  # generated prediction results
├── raw_pssm/                # raw PSSM files used by inference
├── weights/                 # model, tokenizer, and feature-normalization weights
│
├── requirements.txt         # Python package requirements
├── environment.yml          # environment specification
├── OAStest_ab.csv           # antibody names used by the legacy batch workflow
├── data.py                  # protein graph construction
├── main.py
├── model.py                 # GraphTrans model definition
├── oas_prediction.py        # OAS preprocessing helpers
├── predict.py               # checked command-line entry point
├── prediction_Abh.py        # end-to-end inference implementation
├── prediction_all.py        # batch runner for OAS names
├── text.py
├── vj_embedding.py
└── mkdssp                   # DSSP executable for Linux
```

Large neural-network weights and feature datasets are not intended to be committed through ordinary Git. They can be distributed through a GitHub Release or another appropriate research-data repository.

## Requirements

The pipeline is intended for Linux with Python 3.10 or 3.11.

Install the Python packages in `requirements.txt`, then install the matching PyTorch Geometric packages for the selected PyTorch/CUDA version.

```bash
pip install -r requirements.txt
```

Alternatively, `environment.yml` is provided as an environment specification.

The following components are external or platform-sensitive:

- PyTorch and PyTorch Geometric (`torch`, `torch-geometric`, `torch-cluster`, `torch-scatter`)
- IgFold and its model files
- AbLang and its model files
- `abnumber`
- PyMOL with the Python API
- PyRosetta, required for IgFold refinement
- DSSP, exposed here as the executable `mkdssp`
- a PSSM generation workflow; this repository contains PSSM parsing, not a general-purpose PSSM generator

PyRosetta and some model weights may require separate registration or download steps. Do not assume that `pip install -r requirements.txt` installs the complete environment.

## Download Model Weights and Required Resources

Large model weights and resource files are distributed separately through **GitHub Releases** because of their file size.

Download the resource archive from the repository's **Releases** page and extract it directly into the repository root.

After extraction, verify that the following resources exist:

```text
weights/model/fold0.ckpt
weights/model/fold1.ckpt
weights/model/fold2.ckpt
weights/model/fold3.ckpt
weights/model/fold4.ckpt
weights/VJ_tokenizer.pkl
weights/PSSM_repr/AAIMax_PSSM_repr.npy
weights/PSSM_repr/AAIMin_PSSM_repr.npy
weights/ablang_weight/model-weight-heavy/

Epitope_lib/Epitopelib_pdb/
Epitope_lib/Epitope_Feature/epitopelib_DSSP/
Epitope_lib/Epitope_Feature/epitopelib_LLM/
Epitope_lib/Epitope_Feature/epitopelib_PSSM/
Epitope_lib/epitope_new.csv
```

The resulting directory structure should therefore look like:

```text
EpiAbSpec/
├── Epitope_lib/
│   ├── Epitopelib_pdb/
│   ├── Epitope_Feature/
│   │   ├── epitopelib_DSSP/
│   │   ├── epitopelib_LLM/
│   │   └── epitopelib_PSSM/
│   └── epitope_new.csv
│
├── weights/
│   ├── model/
│   │   ├── fold0.ckpt
│   │   ├── fold1.ckpt
│   │   ├── fold2.ckpt
│   │   ├── fold3.ckpt
│   │   └── fold4.ckpt
│   ├── VJ_tokenizer.pkl
│   ├── PSSM_repr/
│   │   ├── AAIMax_PSSM_repr.npy
│   │   └── AAIMin_PSSM_repr.npy
│   └── ablang_weight/
│       └── model-weight-heavy/
└── ...
```

The epitope PDB and feature files must use the same base names.

For reproducibility, we recommend using a versioned Release associated with the corresponding version of the code.

## Input Preparation

Every antibody must have:

```text
oas_fasta/<name>.fasta
oas_VJ/<name>_VJ.csv
raw_pssm/<name>_AbH.pssm
```

### FASTA

The FASTA should contain `H` and `L` records, for example:

```text
>H
HEAVY_CHAIN_SEQUENCE
>L
LIGHT_CHAIN_SEQUENCE
```

### V/J annotation

The V/J CSV must contain:

```text
name
H_Vgene
H_Jgene
H_Species
L_Vgene
L_Jgene
L_Species
```

The current inference code uses the heavy-chain V/J fields.

### PSSM

A PSSM for each antibody must be prepared before prediction:

```text
raw_pssm/<name>_AbH.pssm
```

This repository contains the PSSM parsing functionality required by inference, but does not provide a general-purpose PSSM generation workflow.

## Run One Antibody

All commands should be run from the **repository root**, because the inference implementation uses relative paths.

For an antibody named `ab1`, first verify that these files exist:

```text
oas_fasta/ab1.fasta
oas_VJ/ab1_VJ.csv
raw_pssm/ab1_AbH.pssm
```

Then run:

```bash
python predict.py --name ab1
```

The command checks the local resources and then calls the existing inference implementation.

The result is written to:

```text
output/ab1_pred.csv
```

The output is sorted by descending `Prediction` and contains:

| Column | Description |
| --- | --- |
| `epitope_ID` | epitope structure identifier |
| `antibody_ID` | generated antibody heavy-chain identifier |
| `Prediction` | averaged sigmoid-transformed model score |
| `Epitope - Source Organism` | epitope source metadata |
| `Epitope - Molecule Parent` | parent antigen metadata |

## Batch Prediction

To run the legacy batch workflow over names in `OAStest_ab.csv`:

```bash
python prediction_all.py
```

This may take substantial time and disk space because it creates intermediate antibody and temporary graph-data directories for each sample.

## Reproducibility Notes

- Run all commands from the repository root; the legacy inference implementation uses relative paths.
- Inference automatically selects CUDA when available, otherwise CPU.
- Five checkpoints are loaded and their predictions are averaged.
- `mkdssp` in the repository root is a Linux executable. Use a system DSSP installation on other platforms and update the inference command if needed.
- `raw_pssm` must contain a PSSM for each antibody before prediction.
- Large generated outputs and intermediate files should not be committed to the repository.
