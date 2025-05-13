import os
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd

class Trainer:
    
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, 
                 epochs, device, util, verbose, patience, no_stop):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.util = util
        self.early_stopping = EarlyStopping(verbose, patience, no_stop)
        
    def fit(self, filename, is_chirps=False):
        train_losses, val_losses = [], []

        for epoch in tqdm(range(1,self.epochs+1)):
            train_loss = self.__train(is_chirps)
            evaluator = Evaluator(self.model, self.loss_fn, self.optimizer, self.val_loader, self.device, self.util)
            val_loss,_ = evaluator.eval(is_test=False, is_chirps=is_chirps, epoch=epoch)
            if (self.verbose):
                print(f'Epoch: {epoch}/{self.epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            self.early_stopping(val_loss, self.model, self.optimizer, epoch, filename)
            if (torch.cuda.is_available()):
                torch.cuda.empty_cache()
            if self.early_stopping.isToStop:
                if (self.verbose):
                    print("=> Stopped")
                break

        return train_losses, val_losses

    def __train(self, is_chirps=False):
        self.model.train()
        epoch_loss = 0.0
        mask_land = self.util.get_mask_land().to(self.device)
        for batch_idx, (inputs, target) in enumerate(self.train_loader):
            inputs, target = inputs.to(self.device), target.to(self.device)
            # get prediction
            output = self.model(inputs)
            if is_chirps:
                output = mask_land * output
            loss = self.loss_fn(output, target)
            # clear previous gradients 
            self.optimizer.zero_grad()
            # compute gradients
            loss.backward()
            # performs updates using calculated gradients
            self.optimizer.step()
            epoch_loss += loss.item()

        return  epoch_loss/len(self.train_loader)
            
            
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience
    """
    
    def __init__(self, verbose, patience, no_stop):
        self.verbose = verbose
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.isToStop = False
        self.enable_stop = not no_stop
          
    def __call__(self, val_loss, model, optimizer, epoch, filename):
        is_best = bool(val_loss < self.best_loss)
        if (is_best):
            self.best_loss = val_loss
            self.__save_checkpoint(self.best_loss, model, optimizer, epoch, filename)
            self.counter = 0
        elif (self.enable_stop):
            self.counter += 1
            if (self.verbose):
                print(f'=> Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.isToStop = True
    
    def __save_checkpoint(self, loss, model, optimizer, epoch, filename):
        state = {'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch,
                 'loss': loss}
        torch.save(state, filename)
        if (self.verbose):
            print ('=> Saving a new best') 
        
    
class Evaluator:
        
    def __init__(self, model, loss_fn, optimizer, data_loader, device, util=None, step=0, iteration_number=1):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.util = util
        self.step = int(step)
        self.device = device
        self.iteration_number = iteration_number
       
    def eval(self, is_test=True, is_chirps=False, epoch=1):
        self.model.eval()
        cumulative_rmse, cumulative_mae = 0.0, 0.0
        observation_rmse, observation_mae = [0]*self.step, [0]*self.step
        loader_size = len(self.data_loader)
        mask_land = self.util.get_mask_land().to(self.device)

        weak_threshold = np.log1p(5)      # [0,5)
        moderate_threshold = np.log1p(25) # [5,25)
        heavy_threshold = np.log1p(50)    # [25,50)

        conf_matrix = {
            "0-5": {"0-5": 0, "5-25": 0, "25-50": 0, "50-inf": 0},
            "5-25": {"0-5": 0, "5-25": 0, "25-50": 0, "50-inf": 0},
            "25-50": {"0-5": 0, "5-25": 0, "25-50": 0, "50-inf": 0},
            "50-inf": {"0-5": 0, "5-25": 0, "25-50": 0, "50-inf": 0},
        }

        # level_0_5_true = 0
        # level_5_25_true = 0
        # level_25_50_true = 0
        # level_50_inf_true = 0

        with torch.no_grad(): 
            # print(f"TOTAL BATCHES: {loader_size}")
            
            # print("self.data_loader.dataset.y.shape", self.data_loader.dataset.y.shape)
            # torch.Size([100, 1, 5, 9, 11])

            for batch_i, (inputs, target) in enumerate(self.data_loader):
                # print(inputs.shape) # torch.Size([3, 19, 5, 9, 11]) batch_size = 3
                # print(target.shape) # torch.Size([3, 1, 5, 9, 11]) batch_size = 3
                inputs, target = inputs.to(self.device), target.to(self.device)

                if epoch % 40 == 0 or epoch == 1:
                    target_squeeze = target.squeeze(1)

                    mask_0_5 = target_squeeze < weak_threshold
                    mask_5_25 = (target_squeeze >= weak_threshold) & (target_squeeze < moderate_threshold)
                    mask_25_50 = (target_squeeze >= moderate_threshold) & (target_squeeze < heavy_threshold)
                    mask_50_inf = target_squeeze >= heavy_threshold

                    # level_0_5_true   += mask_0_5.sum().item()
                    # level_5_25_true  += mask_5_25.sum().item()
                    # level_25_50_true += mask_25_50.sum().item()
                    # level_50_inf_true+= mask_50_inf.sum().item()

                output = self.model(inputs)
                if is_chirps:
                    output = mask_land * output    
                rmse_loss = self.loss_fn(output, target)
                mae_loss = F.l1_loss(output, target)
                cumulative_rmse += rmse_loss.item()
                cumulative_mae += mae_loss.item()
                
                if is_test:
                    #metric per observation (lat x lon) at each time step (t) 
                    for i in range(self.step):
                        output_observation = output[:,:,i,:,:]
                        target_observation = target[:,:,i,:,:]
                        rmse_loss_obs = self.loss_fn(output_observation,target_observation)
                        mae_loss_obs = F.l1_loss(output_observation, target_observation)
                        observation_rmse[i] += rmse_loss_obs.item()
                        observation_mae[i] += mae_loss_obs.item()
                
                if epoch % 40 == 0 or epoch == 1:
                    # print(output.shape) # torch.Size([3, 1, 5, 9, 11])
                    # print(output[:, 0, :, :, :].shape) # torch.Size([3, 5, 9, 11])
                    y_pred = output[:, 0, :, :, :]

                    conf_matrix["0-5"]["0-5"] += (
                        (y_pred < weak_threshold) & mask_0_5
                    ).sum().item()
                    conf_matrix["0-5"]["5-25"] += (
                        ((y_pred >= weak_threshold) & (y_pred < moderate_threshold)) & mask_0_5
                    ).sum().item()
                    conf_matrix["0-5"]["25-50"] += (
                        ((y_pred >= moderate_threshold) & (y_pred < heavy_threshold)) & mask_0_5
                    ).sum().item()
                    conf_matrix["0-5"]["50-inf"]+= (
                        (y_pred >= heavy_threshold) & mask_0_5
                    ).sum().item()

                    conf_matrix["5-25"]["0-5"] += (
                        (y_pred < weak_threshold) & mask_5_25
                    ).sum().item()
                    conf_matrix["5-25"]["5-25"] += (
                        ((y_pred >= weak_threshold) & (y_pred < moderate_threshold)) & mask_5_25
                    ).sum().item()
                    conf_matrix["5-25"]["25-50"] += (
                        ((y_pred >= moderate_threshold) & (y_pred < heavy_threshold)) & mask_5_25
                    ).sum().item()
                    conf_matrix["5-25"]["50-inf"] += (
                        (y_pred >= heavy_threshold) & mask_5_25
                    ).sum().item()

                    conf_matrix["25-50"]["0-5"] += (
                        (y_pred < weak_threshold) & mask_25_50
                    ).sum().item()
                    conf_matrix["25-50"]["5-25"] += (
                        ((y_pred >= weak_threshold) & (y_pred < moderate_threshold)) & mask_25_50
                    ).sum().item()
                    conf_matrix["25-50"]["25-50"] += (
                        ((y_pred >= moderate_threshold) & (y_pred < heavy_threshold)) & mask_25_50
                    ).sum().item()
                    conf_matrix["25-50"]["50-inf"] += (
                        (y_pred >= heavy_threshold) & mask_25_50
                    ).sum().item()

                    conf_matrix["50-inf"]["0-5"] += (
                        (y_pred < weak_threshold) & mask_50_inf
                    ).sum().item()
                    conf_matrix["50-inf"]["5-25"] += (
                        ((y_pred >= weak_threshold) & (y_pred < moderate_threshold)) & mask_50_inf
                    ).sum().item()
                    conf_matrix["50-inf"]["25-50"] += (
                        ((y_pred >= moderate_threshold) & (y_pred < heavy_threshold)) & mask_50_inf
                    ).sum().item()
                    conf_matrix["50-inf"]["50-inf"] += (
                        (y_pred >= heavy_threshold) & mask_50_inf
                    ).sum().item()
                        
            if is_test:             
                self.util.save_examples(inputs, target, output, self.step, self.iteration_number)
                print('>>>>>>>>> Metric per observation (lat x lon) at each time step (t)')
                print('RMSE')
                print(*np.divide(observation_rmse, batch_i+1), sep = ",")
                print('MAE')
                print(*np.divide(observation_mae, batch_i+1), sep = ",")
                print('>>>>>>>>')  

        if epoch % 40 == 0 or epoch == 1:
            confusion_df = pd.DataFrame(conf_matrix).T
            print(f"\nConfusion matrix at epoch {epoch}")
            # print(f"batch_i: {batch_i}")
            print(confusion_df)
            # print({
            #     "level_0_5_true": level_0_5_true,
            #     "level_5_25_true": level_5_25_true,
            #     "level_25_50_true": level_25_50_true,
            #     "level_50_inf_true": level_50_inf_true,
            # })
            # print(f"Sum of true values: {level_0_5_true + level_5_25_true + level_25_50_true + level_50_inf_true}")

            # print(f"Sum of confusion_df values: {confusion_df.sum().sum()}")
            # print(f"Sum of predicted values: {sum(confusion_df.sum(axis=1))}")
            # print(f"Sum of conf_matrix: {confusion_df.sum().sum()}")
            # print(f"level_50_inf_true: {level_50_inf_true}")
                
        return cumulative_rmse/loader_size,cumulative_mae/loader_size
        
        
    def load_checkpoint(self, filename, dataset_type=None, model=None):
        if not(os.path.isabs(filename)):
            filename = os.path.join('output', dataset_type, 'checkpoints', model.lower(), filename)  
        epoch, loss = 0.0, 0.0
        if os.path.isfile(filename):
            checkpoint = torch.load(filename)
            name = os.path.basename(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print(f'=> Loaded checkpoint {name} (best epoch: {epoch}, validation rmse: {loss:.4f})')
        else:
            print(f'=> No checkpoint found at {filename}')

        return epoch, loss