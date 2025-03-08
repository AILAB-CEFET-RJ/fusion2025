# Fusion2025: Data Fusion for Precipitation Nowcasting in a Spatiotemporal Context

In this repository we employ data fusion techniques to propose an algorithm that integrates pluviometric data from surface stations with observations from the ERA5 reanalysis model, incorporating data from sirens (Websirens), Weather Stations (INMET) and Rain Gauge Stations (AlertaRio) networks. 


## Requirements

Mainly, our code uses Python 3.6. See [config/environment.yml](Link para o env) for other requirements.

To generate the environment for the experiment, run the code below:

```
./setup.sh
```

## Datasets

All datasets are publicly available at http://doi.org/10.5281/zenodo.3558773 (Substituir pelo Link verdadeiro após publicar Zenodo). Datasets must be placed in the [data](Like para pasta data) folder.

Note: The downloaded files must be decompressed into the specified folder.
<!---
## Experiments

Jupyter notebooks for the first sequence-to-sequence task (given the previous 5 grids, predict the next 5 grids) can be found in the [notebooks](https://github.com/MLRG-CEFET-RJ/stconvs2s/tree/master/notebooks) folder (see Table 1 and 3 in the paper).


Final experiments (see Table 2 in the paper) compare STConvS2S (our architecture) with deep learning state-of-the-art architectures and ARIMA models. We evaluate the models in two horizons: 5 and 15-steps ahead. This task is performed using [main.py](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/main.py) (for deep learning models) and [arima.py](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/arima.py) (for ARIMA models). Results can be found in the [output](https://github.com/MLRG-CEFET-RJ/stconvs2s/tree/master/output) folder.


* `/output/full-dataset` (for deep learning models)
	* `/checkpoints`: pre-trained models that allow you to recreate the training configuration (weights, loss, optimizer).
	* `/losses`: training and validation losses. Can be used to recreate the error analysis plot
	* `/plots`:	error analysis plots from training phase
	* `/results`: evaluation phase results with metric value (RMSE and MAE), training time, best epochs and so on.

* `/output/arima` (for ARIMA models)
	
-->
## Usage

For training and testing models using the different combinations of data sources with the data from ERA5 Reanalysis Model, you can use the following scripts.

### ERA5 - Reanalysis Model (Only)
```
nohup python main.py -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot -dsp "data/output_dataset_era5_only_2011-01_2024-10.nc" -r "full-dataset_200epochs_0.0001LR_1channeltarget_2011-1_2024-10_era5_only_19features" &> train_era5_only_2011_2024_1_channel.log &
```

### Alerta Rio - Rain Gauge Stations
```
nohup python main.py -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot -dsp "data/output_dataset_alertario_only_2011-01_2024-10.nc" -r "full-dataset_200epochs_0.0001LR_1channeltarget_2011-1_2024-10_alertario_only_19features" &> train_alertario_only_2011_2024_1_channel.log &
```

### INMET - Weather Stations
```
nohup python main.py -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot -dsp "data/output_dataset_inmet_only_2011-01_2024-10.nc" -r "full-dataset_200epochs_0.0001LR_1channeltarget_2011-1_2024-10_inmet_only_19features" &> train_inmet_only_2011_2024_1_channel.log &
```

### WebSirens - Sirens
```
nohup python main.py -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot -dsp "data/output_dataset_sirenes_only_2011-01_2024-10.nc" -r "full-dataset_200epochs_0.0001LR_1channeltarget_2011-1_2024-10_sirenes_only_19features" &> train_sirenes_only_2011_2024_1_channel.log &
```

### Alerta Rio - Rain Gauge Stations and INMET - Weather Stations
```
nohup python main.py -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot -dsp "data/output_dataset_inmet_alertario_2011-01_2024-10.nc" -r "full-dataset_200epochs_0.0001LR_1channeltarget_2011-1_2024-10_inmet+alertario_only_19features" &> train_inmet+alertario_only_2011_2024_1_channel.log &
```

### Alerta Rio - Rain Gauge Stations and WebSirens - Sirens
```
nohup python main.py -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot -dsp "data/output_dataset_websirenes_alertario_2011-01_2024-10.nc" -r "full-dataset_200epochs_0.0001LR_1channeltarget_2011-1_2024-10_sirenes+alertario_19features" &> train_sirenes+alertario_2011_2024_1_channel.log &
```

### INMET - Weather Stations and WebSirens - Sirens
```
nohup python main.py -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot -dsp "data/output_dataset_websirenes_inmet_2011-01_2024-10.nc" -r "full-dataset_200epochs_0.0001LR_1channeltarget_2011-1_2024-10_sirenes+inmet_19features" &> train_sirenes+inmet_2011_2024_1_channel.log &
```

### Alerta Rio - Rain Gauge Stations, INMET - Weather Stations and WebSirens - Sirens
```
nohup python main.py -i 1 -v 4 -m stconvs2s-r -e 200 -p 100 --plot -dsp "data/output_dataset_websirenes_inmet_alertario_2011-01_2024-10.nc" -r "full-dataset_200epochs_0.0001LR_1channeltarget_2011-1_2024-10_sirenes+inmet+alertario_19features" &> train_sirenes+inmet+alertario_2011_2024_1_channel.log &
```

All experiment can be executed automatically using the following shell script:
```
./run_models.sh
```

<!---
Check out the other possible parameters [here](https://github.com/MLRG-CEFET-RJ/stconvs2s/blob/master/main.py#L15-L34).

 ## Citation
```
@article{Castro2021,
	author 	= "Rafaela Castro and Yania M. Souto and Eduardo Ogasawara and Fabio Porto and Eduardo Bezerra",
	title 	= "STConvS2S: Spatiotemporal Convolutional Sequence to Sequence Network for Weather Forecasting",
	journal = "Neurocomputing",
	volume 	= "426",
	pages 	= "285 - 298",
	year 	= "2021",
	issn 	= "0925-2312",
	doi 	= "https://doi.org/10.1016/j.neucom.2020.09.060"
}
```-->

<!--- ## Contact
To give your opinion about this work, send an email to `rafaela.nascimento@eic.cefet-rj.br`.-->
