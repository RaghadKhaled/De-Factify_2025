#  Robust Synthetic Image Detection (RSID)

## Beyond RGB: Exploring Alternative Color Spaces for Robust Synthetic Image Detection (AAAI 2025)

[[Paper](https://)] [[HuggingFace Model](https://)]

## Architecture figure ##


## Contents

- [Overview](#overview)
- [Setup](#environment-setup)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained model](#pretrained-model)
- [Citation](#acknowledgments)
- [Acknowledgments](#acknowledgments)



## Overview
This repository will contain the official code for the **Beyond RGB: Exploring Alternative Color Spaces for Robust Synthetic Image Detection** submission to **Fourth Workshop on Multimodal Fact-Checking and Hate Speech Detection** Workshop 2025 at AAAI. Our project is organized into three main folders, ...


## Environment Setup
You can set up the Conda environment to get you up and running:
```bash

conda create -n RSID python=3.10 -y
conda activate RSID

git clone https://github.com/RaghadKhaled/De-Factify_2025.git

cd De-Factify_2025

pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
 
pip install -r requirements.txt

```

## Datasets

You can download the row data from:
- Training dataset from [[HuggingFace](https://huggingface.co/datasets/NasrinImp/Defactify4_Train)].
- Validation dataset from [[HuggingFace](https://huggingface.co/datasets/NasrinImp/Defactify4_Validation)].
- Custom benchmark from [[Google Drive](https://drive.google.com/drive/folders/1DgiN4aeTbEdHt9Pre_iQxfVn_KOEhXlJ?usp=drive_link)].


## Training
- You can train the model by running following command:
```bash
sh train.sh
```

The parameters values in `train.sh` file are as following:

- `A`: ..... (default: 2).
- `B`: '...'.
- `C`: file .py for the first approach and file .py for the second approach.

## Evaluation

### Rosbutness evaluation

- You can evaluate the model by running the following command:
```bash
sh evaluate.sh
```

The parameters values in `evaluate.sh` file are as following:

- `A`: ..... (default: 2).
- `B`: '...'.
- `C`: file .py for the image-text strategy, file .py for the NN strategy, and file .py for the RF strategy.

### Generalization evaluation
.....

## Pretrained model

- Download all pretrained weight files from [[here](https://drive.google.com/drive/folders/1DgiN4aeTbEdHt9Pre_iQxfVn_KOEhXlJ?usp=drive_link)].


## Citation
If you use this code/dataset for your research, please cite the reference:
```bash
BibTexX format
```

## Acknowledgments
Our code is heavily based on [[CLIP](https://github.com/openai/CLIP)] and [[LASTED](https://github.com/HighwayWu/LASTED)] We thank the authors for open-sourcing their code.
 
