# Fusion 2025

Code for the paper "Towards a Spatiotemporal Fusion Approach to Precipitation Nowcasting"

Link 1 - 

Link 2 - 

## Abstract

With the increasing availability of meteorological data from various sensors, numerical models and reanalysis products, the need for efficient data integration methods has become paramount for improving weather forecasts and hydrometeorological studies. In this work, we propose a data fusion approach for precipitation nowcasting by integrating data from meteorological and rain gauge stations in Rio de Janeiro metropolitan area with ERA5 reanalysis data and GFS numerical weather prediction. We employ the spatiotemporal deep learning architecture called STConvS2S, leveraging a structured dataset covering a 9 x 11 grid. The study spans from January 2011 to October 2024, and we evaluate the impact of integrating three surface station systems. Among the tested configurations, the fusion-based model achieves an F1-score of 0.2033 for forecasting heavy precipitation events (greater than 25 mm/h) at a one-hour lead time. Additionally, we present an ablation study to assess the contribution of each station network and propose a refined inference strategy for precipitation nowcasting, integrating the GFS numerical weather prediction (NWP) data with in-situ observations.

## Replicating the training experiments

In this step-by-step guide, we'll replicate the ERA5+SIA experiment, which is the experiment using the STConvS2S deep learning architecture with the **S**irenes, **I**nmet, **A**lertaRio and ERA5 reanalysis data integrated into a data fusion model.

1- Clone the repository

```sh
mkdir fusion2025-spatiotemporal-precipitation-nowcasting
cd fusion2025-spatiotemporal-precipitation-nowcasting
git clone https://github.com/AILAB-CEFET-RJ/fusion2025.git .
```

2- Download the datasets from [zenodo](zenodo). For example, we'll download ERA5+SIA dataset:

3- Place the downloaded dataset in the `STConvS2s/data` folder.
```sh
mv <path_to_downloaded_dataset> fusion2025-spatiotemporal-precipitation-nowcasting/STConvS2s/data/
```

4- Clone and activate the conda environment:
```sh
conda env create -f environment.yml -n atmoseer
conda activate atmoseer
```

4- Evaluate the model with the MAE error, Bias, Confusion Matrix and F1-Score metrics.

```sh
 python -m evaluate_training.main --dataset "/home/user/ERA5+IA.nc" --model "/home/user/fusion2025/trained_models/ERA5+IA/checkpoints/stconvs2s-r/cfsr_step5_4_20250513-184502.pth.tar"
```

Expected output:
```sh
Running evaluation with args:
dataset: /home/felipe/rafaela-model/rafaela_model/models_evaluation/output_dataset_replaced-ERA5+IA.nc
model: /home/felipe/fusion2025/trained_models/RunModels_replaced_ERA5+IA_camera-ready/checkpoints/stconvs2s-r/cfsr_step5_4_20250513-184502.pth.tar
lead_time: 1

# ...

Rain Distribution (True): {'light': 52127, 'moderate': 1364, 'heavy': 168, 'extreme': 32}
Rain Distribution (Pred): {'light': 51768, 'moderate': 1770, 'heavy': 135, 'extreme': 18}

# ...

MAE: {'0-5': 0.39043482214408665, '5-25': 7.104901246311378, '25-50': 19.30523035072145, '50-inf': 37.6655695438385}
BIAS: {'0-5': 0.2547201734437317, '5-25': -2.8752870542213014, '25-50': -14.766394019126892, '50-inf': -36.00741457939148}

# ...
          0-5  5-25  25-50  50-inf
0-5     51033  1066     25       3
5-25      692   616     52       4
25-50      40    75     45       8
50-inf      3    13     13       3
unique weights: [2.57500911e-01 9.84072581e+00 7.98973214e+01 4.19460938e+02]
F1 Score for 0-5: 0.9824
F1 Score for 5-25: 0.3931
F1 Score for 25-50: 0.2970
F1 Score for 50-inf: 0.1200
f1_per_class_T+1.pkl written
    Writting f1_score_weighted.pkl:
    ./models_evaluation/output_dataset_replaced-ERA5+IA.nc/f1_score_weighted.pkl
    f1_weighted=0.3899858740214056


    Writting f1_score_macro.pkl:
    ./models_evaluation/output_dataset_replaced-ERA5+IA.nc/f1_score_macro.pkl
    f1_macro=0.4481333100795569
```

This script will also generate the following files during evaluation, which provides a model performance visualization and a serialized model report, for later analysis and comparison with other models:
```sh
models_evaluation
   └── output_dataset_replaced-ERA5+IA.nc
        ├── [0-5)_confusion_matrix.png
        ├── [25-50)_confusion_matrix.png
        ├── [5-25)_confusion_matrix.png
        ├── [50-inf)_confusion_matrix.png
        ├── bias.pkl
        ├── bias_per_category.png
        ├── confusion_matrix.tex
        ├── f1_per_class.pkl
        ├── f1_per_class_T+1.pkl
        ├── f1_score_macro.pkl
        ├── f1_score_weighted.pkl
        ├── rain_distribution.png
        ├── rain_distribution_histogram.png
        ├── rain_distribution_true_pred.png
        ├── mae.pkl
        └── mae_per_category.png
```

<hr>

There's a shell script `STConvS2s/run_models.sh` that help us to run the training script by passing the eight different datasets sequentially, as presented in the table below.

<!-- table with different datasets -->

## Replicating the inference experiments

trazer o script e o dataset irmaõ, vai dar certo em nome de JESUS!
