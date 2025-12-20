# Retrieval-based Text-augmented Multivariate Time Series Forecasting

> This repository contains the code for the paper submitted to ICLR 2026, which proposes a new method for multimodal multivariate time series forecasting.

---

## Usage

### 1. Prepare Dataset

1. Download **Time-MMD** (https://github.com/AdityaLab/Time-MMD) and put it at `./dataset/`
2. Run preprocessing scripts for the specific dataset in `./pre_process/Time-MMD/`, including:

- `NumericalPackage.py`  
- `TextPackage.py`  
- `ChannelDescription_Pack.py`  
- `NewsDatabasePre_embed.py`  
- `Text_preembedding` (run twice: `--ForecastingPoint 0` for channel description, `--ForecastingPoint 1` for query texts)

---

### 2. Train and Evaluation

Run the training and evaluation script for the specific dataset:

```bash
bash ./run_XXX.sh  #(Replace XXX with the dataset name.)