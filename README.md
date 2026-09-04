# EpiAbSpec antibody-epitope prediction

This repository contains an antibody-epitope binding prediction pipeline based on protein structural features and a Graph Transformer model.

Given an antibody identifier with its FASTA and V/J annotation, the pipeline predicts a score for every epitope in `Epitope_lib` and writes a ranked CSV file.


## Pipeline

```text
OAS CSV / antibody FASTA + V/J annotation
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

The model currently uses the antibody heavy chain for the graph and V/J features. The light-chain sequence is accepted by the FASTA reader and passed to IgFold, but light-chain features are not used by the trained predictor.

## Repository layout

```text
prediction/
├── prediction_Abh.py       # end-to-end inference implementation
├── predict.py               # checked command-line entry point
├── data.py                  # protein graph construction
├── model.py                 # GraphTrans model definition
├── oas_prediction.py        # OAS preprocessing helpers
├── prediction_all.py        # batch runner for OAS names
├── result_calu.py           # evaluation of ranked predictions
├── Epitope_lib/             # epitope structures, features and metadata
├── weights/                 # model, tokenizer and feature-normalization weights
├── oas_fasta/               # antibody FASTA inputs
├── oas_VJ/                  # antibody V/J annotations
├── raw_pssm/                # raw PSSM files used by inference
└── output/                  # generated predictions; do not commit bulk results
```

Large structure/feature datasets and neural-network weights should be distributed through Git LFS, a GitHub Release, or an institutional data repository. Ordinary GitHub pushes reject files larger than 100 MB.

## Requirements

The pipeline is intended for Linux with Python 3.10 or 3.11. Install the Python packages in `requirements.txt`, then install the matching PyTorch Geometric packages for the selected PyTorch/CUDA version.

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

## Required resources

Before inference, verify these resources exist:

```text
weights/model/fold0.ckpt ... fold4.ckpt
weights/VJ_tokenizer.pkl
weights/PSSM_repr/AAIMax_PSSM_repr.npy
weights/PSSM_repr/AAIMin_PSSM_repr.npy
weights/ablang_weight/model-weight-heavy/
Epitope_lib/Epitopelib_pdb/
Epitope_lib/Epitope_Feature/epitopelib_DSSP/
Epitope_lib/Epitope_Feature/epitopelib_LLM/
Epitope_lib/Epitope_Feature/epitopelib_PSSM/
Epitope_lib/epitope_new.csv
raw_pssm/<name>_AbH.pssm
```

The epitope PDB and feature files must use the same base names. Every antibody must have both:

```text
oas_fasta/<name>.fasta
oas_VJ/<name>_VJ.csv
```

The FASTA should contain `H` and `L` records, for example:

```text
>H
HEAVY_CHAIN_SEQUENCE
>L
LIGHT_CHAIN_SEQUENCE
```

The V/J CSV must contain `name`, `H_Vgene`, `H_Jgene`, `H_Species`, `L_Vgene`, `L_Jgene`, and `L_Species` columns. The current inference code uses the heavy-chain V/J fields.

## Run one antibody

Run commands from the repository root:

```bash
python predict.py --name ab1
```

The command checks the local resources and then calls the existing inference implementation. The result is written to:

```text
output/ab1_pred.csv
```

The output is sorted by descending `Prediction` and contains:

- `epitope_ID`: epitope structure identifier
- `antibody_ID`: generated antibody heavy-chain identifier
- `Prediction`: averaged sigmoid-transformed model score
- `Epitope - Source Organism`: epitope source metadata
- `Epitope - Molecule Parent`: parent antigen metadata

To run the legacy batch workflow over names in `OAStest_ab.csv`:

```bash
python prediction_all.py
```

This may take substantial time and disk space because it creates intermediate antibody and temporary graph-data directories for each sample.

## Reproducibility notes

- Run from the repository root; the legacy inference implementation uses relative paths.
- Inference automatically selects CUDA when available, otherwise CPU.
- Five checkpoints are loaded and their predictions are averaged.
- `mkdssp` in this directory is a Linux executable. Use a system DSSP installation on other platforms and update the inference command if needed.
- `raw_pssm` must contain a PSSM for each antibody before prediction.
