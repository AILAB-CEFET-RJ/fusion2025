# Fusion 2025

Code for the paper "Towards a Spatiotemporal Fusion Approach to Precipitation Nowcasting"

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

4- Evaluate the model with the RMSE, Bias, Confusion Matrix and F1-Score metrics.

```sh
python STConvS2s/evaluate.py --model_path <path_to_model> --data_path <path_to_data> --metrics rmse bias confusion f1
```

<hr>

There's a shell script `STConvS2s/run_models.sh` that help us to run the training scripts passing the eight different datasets, as presented in the table below.

<!-- table with different datasets -->

## Replicating the inference experiments

trazer o script e o dataset irmaõ, vai dar certo em nome de JESUS!
