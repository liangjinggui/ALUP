# ALUP

This the code for the paper: [Actively Learn from LLMs with Uncertainty Propagation for Generalized Category Discovery](https://aclanthology.org/2024.naacl-long.434.pdf), please check the paper for more details.

## Data

We performed experiments on three public datasets: banking, clinc and stackoverflow, which have been included in our repository in the data folder '[./data](./data/)'.

## Requirements
- python==3.10
- pytorch==2.0.1
- transformers==4.29.2
- numpy==1.25.2
- faiss==1.7.2

## Pretrain Model Preparation

Get the pre-trained BERT model by running:
```bash
git clone https://huggingface.co/bert-base-uncased ./pretrained_models/bert
```

## Running

**1. Pretraining**
```bash
sh scripts/alup/run_pretrain_$dataset$.sh
```

**2. ALUP Turning**
```bash
sh scripts/alup/run_alup_$dataset$.sh 
```

## Note

We include the LLM-generated outputs in the repository, which can be found in the folder '[./outs/alup](./outs/alup)'.

## Thanks

Some code and instructions reference the following repositories:
- [DeepAligned-Clustering](https://github.com/thuiar/DeepAligned-Clustering)
- [TEXTOIR](https://github.com/thuiar/TEXTOIR)
- [MTPCLNN](https://github.com/fanolabs/NID_ACLARR2022)
- [DWGF](https://github.com/yibai-shi/DWGF)
