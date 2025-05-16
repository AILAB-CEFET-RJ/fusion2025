# Fusion 2025

Code for the paper "Towards a Spatiotemporal Fusion Approach to Precipitation Nowcasting"

<img src="./.github/fusion2025-repo-image.png" alt="Fusion 2025" height="400px"/>

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

4- Download the trained model from zenodo and place it in the `STConvS2s/trained_models` folder:

Optionally, instead of downloading the already trained model from zenodo, we may execute the STConvS2S train script with the instructions below:
```sh
python -m STConvS2S.main --cuda 1 -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot --dropout 0.5 -dsp "data/ERA5+SIA.nc"" -r "RunModels_ERA5+SIA"
```

4- Evaluate the model with the MAE error, Bias, Confusion Matrix and F1-Score metrics.

```sh
python -m evaluate_training.main --dataset "/home/user/ERA5+SIA.nc" --model "/home/user/fusion2025/trained_models/ERA5+SIA-model.pth.tar"
```

Expected output:
```sh
Running evaluation with args:
dataset: /home/user/ERA5+SIA.nc
model: /home/user/fusion2025/trained_models/ERA5+SIA-model.pth.tar
lead_time: 1

# ...

Rain Distribution (True): {'light': 51574, 'moderate': 1865, 'heavy': 207, 'extreme': 45}
Rain Distribution (Pred): {'light': 50866, 'moderate': 2620, 'heavy': 180, 'extreme': 25}

# ...

MAE: {'0-5': 0.44631323257810807, '5-25': 6.687025476972155, '25-50': 20.360103851355216, '50-inf': 42.588324568006726}
BIAS: {'0-5': 0.24514703574449076, '5-25': -1.57113556580633, '25-50': -18.280988034418815, '50-inf': -42.02639405992296}

# ...
          0-5  5-25  25-50  50-inf
0-5     50032  1492     45       5
5-25      785   980     85      15
25-50      47   119     38       3
50-inf      2    29     12       2
unique weights: [2.60261954e-01 7.19718499e+00 6.48442029e+01 2.98283333e+02]
F1 Score for 0-5: 0.9768
F1 Score for 5-25: 0.4370
F1 Score for 25-50: 0.1964
F1 Score for 50-inf: 0.0571
f1_per_class_T+1.pkl written
    Writting f1_score_weighted.pkl:
    ./models_evaluation/output_dataset_replaced-ERA5+SIA.nc/f1_score_weighted.pkl
    f1_weighted=0.35905292925986854

    ./models_evaluation/output_dataset_replaced-ERA5+SIA.nc/f1_score_macro.pkl
    f1_macro=0.41683587109105374
```

This script will also generate the following files during evaluation, which provides a model performance visualization and a serialized model report, for later analysis and comparison with other models:
```sh
models_evaluation
   └── ERA5+SIA.nc
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

This section replicates the inference experiments using the ERA5+SIA trained model with the GFS NWP model integrated with AlertaRio (`GFS+A`) dataset.

```sh
python -m evaluate_inference.main --dataset /home/user/GFS+A_dataloader.pt
```

Expected output in stdout:
```sh
Loading inference dataset from ./GFS+A_dataloader.pt
max output:  tensor(36.1225)
min output:  tensor(0.)
Saved figure to ./inference_grid_figures/iteration_0/tp_figure.png
Confusion matrix:
\begin{tabular}{lrrrr}
\toprule
 & 0-5 & 5-25 & 25-50 & 50-inf \\
\midrule
0-5 & 6 & 4 & 1 & 0 \\
5-25 & 0 & 2 & 1 & 0 \\
25-50 & 0 & 1 & 0 & 0 \\
50-inf & 0 & 0 & 0 & 0 \\
\bottomrule
\end{tabular}
```

## Atmoseer

The code used to create the datasets can found at the [Atmoseer package](https://github.com/AILAB-CEFET-RJ/atmoseer/tree/main/src/spatiotemporal_builder/). It is part of the Artificial Intelligence Laboratory (AI-Lab) at CEFET/RJ, maintained by the authors of this paper.
