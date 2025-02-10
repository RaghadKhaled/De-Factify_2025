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
This repository will contain the official code for the **Beyond RGB: Exploring Alternative Color Spaces for Robust Synthetic Image Detection** submission to **Fourth Workshop on Multimodal Fact-Checking and Hate Speech Detection** Workshop 2025 at AAAI.


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

- For the first approach, use `train_first_approach.py` and for the second approach, use `train_second_approach.py`.
- `train_file`: txt file that contains the train img pathes along with the labels. For the first approach use `/annotations/train_set/first_approach.txt` and `/annotations/train_set/second_approach.txt` for the second approach.
- `val_file`: txt file that contains the val img pathes along with the labels. For the first approach use `/annotations/val_set/first_approach.txt` and `/annotations/val_set/second_approach.txt` for the second approach.
- `batch_size`: The training batch size. (default: 48).
- `img_encoder`: The type of image encoder. we use `RN50x64`.
- `data_size`: The image size for training. `448` for the `RN50x64`.
- `lr`: The initial learning rate. (default: 1e-5).
- `epoches`: The training epoches. (default: 100).
- `num_class`: The class number of training dataset. (default: 6).
- `text_option`: `1` for the first approach and `2` for the second approach.
- `color_space`: The type of color spaces. `RGB` or `Lab` or `YCbCr`.

## Evaluation

### Rosbutness evaluation

- You can evaluate the model by running the following command:
```bash
sh eval_robsutness.sh
```

The parameters values in `eval_robsutness.sh` file are as following:

- For the image-text strategy use `eval_robsutness_image_text.py`, for the NN strategy use `eval_robsutness_NN.py` and for the RF strategy use `eval_robsutness_RF.py`.
- `test_files_pathes`: txt file that contain the img pathes.
- `batch_size`: The testing batch size. (default: 48).
- `img_encoder`: The type of image encoder. we use `RN50x64`.
- `data_size`: The image size for training. `448` for the `RN50x64`.
- `num_class`: The class number of testing dataset. (default: 6).
- `text_option`: `1` for the first approach and `2` for the second approach.
- `method`: The type of model. `baseline' or  `RGB` or `Lab` or `YCbCr`.
- `task`: `A` for binary classification, `B` for multi-class classification.
- `transformation`: The type of the testing dataset. `without` means the original images. `with` means the images that undergoes the perturbations.
- `ckps`: `The path for the pre-trained model.
- `task`: `A` for binary classification, `eval_robsutness_NN.py`B` for multi-class classification.


For `eval_robsutness_NN.py`, add the paths for the featues bank using `real_template`, `SD21_template`, `SDXL_template`, `SD3_template`, `Dalle_template`, `Mid_template` parameters.

For `eval_robsutness_RF.py`, add the path for the features used to train the classifier using the `train_path` parameter.



### Generalization evaluation
TBD

## Pretrained model

- Download all pretrained weight files from [[here](https://drive.google.com/drive/folders/1DgiN4aeTbEdHt9Pre_iQxfVn_KOEhXlJ?usp=drive_link)].


## Citation
If you use this code/dataset for your research, please cite the reference:
```bash
TBD
```

## Acknowledgments
Our code is heavily based on [[CLIP](https://github.com/openai/CLIP)] and [[LASTED](https://github.com/HighwayWu/LASTED)] We thank the authors for open-sourcing their code.
 
