import numpy as np
import random as rd
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import time as tm
import os

from .model.stconvs2s import STConvS2S_R, STConvS2S_C
from .model.baselines import *
from .model.ablation import *
 
from .tool.train_evaluate import Trainer, Evaluator
from .tool.dataset import NetCDFDataset
from .tool.loss import RMSELoss, MAELoss
from .tool.utils import Util

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

def clean_precipitation_data(data, property, threshold=0.01, extreme_threshold=150.0, verbose=True):
    # --- PART 1: Remove extreme precipitation values ---
    extreme_mask = data[:, :, :, :, 0] > extreme_threshold
    total_extremes = extreme_mask.sum()
    max_extreme = 0
    if total_extremes > 0:
        max_extreme = data[:, :, :, :, 0][extreme_mask].max()
    data[:, :, :, :, 0][extreme_mask] = 0
    
    # --- PART 2: Clean only middle timesteps (T1, T2, T3) ---
    total_changes_middle = 0
    max_changed_value_middle = 0
    
    # Only handle middle timesteps (T1, T2, T3)
    for t in range(1, data.shape[1] - 1):
        mask = (data[:, t-1, :, :, 0] == 0) & (data[:, t+1, :, :, 0] == 0) & (data[:, t, :, :, 0] > threshold)
        
        changes_at_t = mask.sum()
        total_changes_middle += changes_at_t
        
        if changes_at_t > 0:
            max_val_at_t = data[:, t, :, :, 0][mask].max()
            max_changed_value_middle = max(max_changed_value_middle, max_val_at_t)
        
        data[:, t, :, :, 0][mask] = 0
    
    if verbose:
        print(f"=== Extreme Precipitation Removal ({property}) ===")
        print(f"Total extreme values (>{extreme_threshold} mm/h) removed: {total_extremes}")
        print(f"Percentage of data removed: {100 * total_extremes / data.size:.6f}%")
        print(f"Maximum extreme value: {max_extreme}")
        
        print("\n=== Spurious Precipitation Removal ===")
        print(f"Total spurious values removed: {total_changes_middle}")
        print(f"Percentage of data changed: {100 * total_changes_middle / data.size:.4f}%")
        print(f"Maximum value changed: {max_changed_value_middle}")
    
    return data

class MLBuilder:

    def __init__(self, config, device):
        
        self.config = config
        self.device = device


        # full-dataset_200epochs_0.0001LR_1channeltarget_2011-1_2024-10_websirenesandera5+inmet+alertario_19features-NO-OVERLAPPING-dropout0.25

        # self.dataset_type = 'small-dataset' if (self.config.small_dataset) else 'full-dataset'

        self.dataset_type = 'small-dataset' if (self.config.small_dataset) else config.run_name
        os.environ["run_name"] = self.dataset_type

        self.step = str(config.step)
        self.dataset_name, self.dataset_file = self.__get_dataset_file()
        self.dropout_rate = self.__get_dropout_rate()
        self.filename_prefix = self.dataset_name + '_step' + self.step
                
    def run_model(self, number):
        self.__define_seed(number)
        validation_split = 0.2
        test_split = 0.2
        # Loading the dataset
        ds = xr.open_mfdataset(self.dataset_file).load()
        if (self.config.small_dataset):
            ds = ds[dict(sample=slice(0,500))]

        clean_precipitation_data(ds.x.values, 'x', threshold=0.0, verbose=True)
        clean_precipitation_data(ds.y.values, 'y', threshold=0.0, verbose=True)

        precipitation_x = ds.x.sel(channel=0)
        ds["x"].loc[{"channel": 0}] = np.log1p(precipitation_x)
        print(f"Max precipitation_x: {precipitation_x.max().values}")

        precipitation_y = ds.y.sel(channel=0)
        ds["y"].loc[{"channel": 0}] = np.log1p(precipitation_y)
        print(f"Max precipitation_y: {precipitation_y.max().values}")

        train_dataset = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split)
        val_dataset   = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split, is_validation=True)
        test_dataset  = NetCDFDataset(ds, test_split=test_split, 
                                      validation_split=validation_split, is_test=True)
        if (self.config.verbose):
            print('[X_train] Shape:', train_dataset.X.shape)
            print('[y_train] Shape:', train_dataset.y.shape)
            print('[X_val] Shape:', val_dataset.X.shape)
            print('[y_val] Shape:', val_dataset.y.shape)
            print('[X_test] Shape:', test_dataset.X.shape)
            print('[y_test] Shape:', test_dataset.y.shape)
            print(f'Train on {len(train_dataset)} samples, validate on {len(val_dataset)} samples')
                        
        params = {'batch_size': self.config.batch, 
                  'num_workers': self.config.workers, 
                  'worker_init_fn': self.__init_seed}

        cyan_color = '\033[96m'
        reset_color = '\033[0m'
        print(f"""
            {cyan_color}
            Data Shapes:
            X_train: {train_dataset.X.shape}
            y_train: {train_dataset.y.shape}

            X_val: {val_dataset.X.shape}
            y_val: {val_dataset.y.shape}

            X_test: {test_dataset.X.shape}
            y_test: {test_dataset.y.shape}
            {reset_color}
        """)

        train_loader = DataLoader(dataset=train_dataset, shuffle=True,**params)
        val_loader = DataLoader(dataset=val_dataset, shuffle=False,**params)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, **params)
        
        models = {
            'stconvs2s-r': STConvS2S_R,
            'stconvs2s-c': STConvS2S_C,
            'convlstm': STConvLSTM,
            'predrnn': PredRNN,
            'mim': MIM,
            'conv2plus1d': Conv2Plus1D,
            'conv3d': Conv3D,
            'enc-dec3d': Endocer_Decoder3D,
            'ablation-stconvs2s-nocausalconstraint': AblationSTConvS2S_NoCausalConstraint,
            'ablation-stconvs2s-notemporal': AblationSTConvS2S_NoTemporal,
            'ablation-stconvs2s-r-nochannelincrease': AblationSTConvS2S_R_NoChannelIncrease,
            'ablation-stconvs2s-c-nochannelincrease': AblationSTConvS2S_C_NoChannelIncrease,
            'ablation-stconvs2s-r-inverted': AblationSTConvS2S_R_Inverted,
            'ablation-stconvs2s-c-inverted': AblationSTConvS2S_C_Inverted,
            'ablation-stconvs2s-r-notfactorized': AblationSTConvS2S_R_NotFactorized,
            'ablation-stconvs2s-c-notfactorized': AblationSTConvS2S_C_NotFactorized
        }
        if not(self.config.model in models):
            raise ValueError(f'{self.config.model} is not a valid model name. Choose between: {models.keys()}')
            quit()
            
        # Creating the model    
        model_bulder = models[self.config.model]
        model = model_bulder(train_dataset.X.shape, self.config.num_layers, self.config.hidden_dim, 
                             self.config.kernel_size, self.device, self.dropout_rate, int(self.step))
        model.to(self.device)
        criterion = MAELoss()
        opt_params = {'lr': 0.00001, 
                      'alpha': 0.9, 
                      'eps': 1e-6}
        print(f"opt_params", opt_params)
        optimizer = torch.optim.RMSprop(model.parameters(), **opt_params)
        util = Util(self.config.model, self.dataset_type, self.config.version, self.filename_prefix)
        
        train_info = {'train_time': 0}
        if self.config.pre_trained is None:
            train_info = self.__execute_learning(model, criterion, optimizer, train_loader,  val_loader, util) 
                                                 
        eval_info = self.__load_and_evaluate(model, criterion, optimizer, test_loader, 
                                             train_info['train_time'], util)

        if (torch.cuda.is_available()):
            torch.cuda.empty_cache()

        return {**train_info, **eval_info}


    def __execute_learning(self, model, criterion, optimizer, train_loader, val_loader, util):
        checkpoint_filename = util.get_checkpoint_filename()    
        trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, self.config.epoch, 
                          self.device, util, self.config.verbose, self.config.patience, self.config.no_stop)
    
        start_timestamp = tm.time()
        # Training the model
        train_losses, val_losses = trainer.fit(checkpoint_filename, is_chirps=self.config.chirps)
        end_timestamp = tm.time()
        # Learning curve
        util.save_loss(train_losses, val_losses)
        util.plot([train_losses, val_losses], ['Training', 'Validation'], 'Epochs', 'Loss',
                  'Learning curve - ' + self.config.model.upper(), self.config.plot)

        train_time = end_timestamp - start_timestamp       
        print(f'\nTraining time: {util.to_readable_time(train_time)} [{train_time}]')
               
        return {'dataset': self.dataset_name,
                'dropout_rate': self.dropout_rate,
                'train_time': train_time
                }
                
    
    def __load_and_evaluate(self, model, criterion, optimizer, test_loader, train_time, util):  
        evaluator = Evaluator(model, criterion, optimizer, test_loader, self.device, util, self.step, self.iteration_number)
        if self.config.pre_trained is not None:
            # Load pre-trained model
            best_epoch, val_loss = evaluator.load_checkpoint(self.config.pre_trained, self.dataset_type, self.config.model)
        else:
            # Load model with minimal loss after training phase
            checkpoint_filename = util.get_checkpoint_filename() 
            best_epoch, val_loss = evaluator.load_checkpoint(checkpoint_filename)
        
        time_per_epochs = 0
        if not(self.config.no_stop): # Earling stopping during training
            time_per_epochs = train_time / (best_epoch + self.config.patience)
            print(f'Training time/epochs: {util.to_readable_time(time_per_epochs)} [{time_per_epochs}]')
        
        test_rmse, test_mae = evaluator.eval(is_chirps=self.config.chirps)
        print(f'Test RMSE: {test_rmse:.4f}\nTest MAE: {test_mae:.4f}')
                        
        return {'best_epoch': best_epoch,
                'val_rmse': val_loss,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'train_time_epochs': time_per_epochs
                }
          
    def __define_seed(self, number):      
        self.iteration_number = number
        if (~self.config.no_seed):
            # define a different seed in every iteration 
            seed = (number * 10) + 1000
            np.random.seed(seed)
            rd.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic=True
            
    def __init_seed(self, number):
        seed = (number * 10) + 1000
        np.random.seed(seed)
        
    def __get_dataset_file(self):
        dataset_file, dataset_name = None, None
        if (self.config.chirps):
            # data/output_dataset_2011-01_2024-10.nc IS ONLY WEBSIRENES OKAY? I renamed the file to output_dataset_websirenes_2011-01_2024-10.nc
            # dataset_file = 'data/output_dataset_2011-01_2024-10.nc'
            # dataset_file = 'data/output_dataset_brinquedo.nc'
            # dataset_file = "data/output_dataset_websirenes+inmet_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes+inmet+alertario_2011-01_2024-10.nc"

            # dataset_file = "data/output_dataset_era5_only_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes+inmet_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes+inmet+alertario_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes+inmet+alertario_2011-01_2024-10_no_overlapping.nc"
            # dataset_file = "data/output_dataset_alertario_only_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_inmet_only_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes+alertario_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_inmet+alertario_2011-01_2024-10.nc"
            dataset_name = 'chirps'
        else:
            # dataset_file = 'data/output_dataset_2011-01_2024-10.nc'
            # dataset_file = 'data/output_dataset_brinquedo.nc'
            # dataset_file = "data/output_dataset_websirenes+inmet_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes+inmet+alertario_2011-01_2024-10.nc"

            # dataset_file = "data/output_dataset_era5_only_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes+inmet_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes+inmet+alertario_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes+inmet+alertario_2011-01_2024-10_no_overlapping.nc"
            # dataset_file = "data/output_dataset_alertario_only_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_inmet_only_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_websirenes+alertario_2011-01_2024-10.nc"
            # dataset_file = "data/output_dataset_inmet+alertario_2011-01_2024-10.nc"
            dataset_name = 'cfsr'

        dataset_file = self.config.dataset_path
        if dataset_file is None:
            print("deu ruim no dataset felipe, passar -dsp com o path")
            exit(0)
        
        return dataset_name, dataset_file
        
    def __get_dropout_rate(self):
        dropout_rates = {
            'predrnn': 0.5,
            'mim': 0.5
        }
        if self.config.model in dropout_rates:
            dropout_rate = dropout_rates[self.config.model] 
        else:
            dropout_rate = 0.

        return self.config.dropout