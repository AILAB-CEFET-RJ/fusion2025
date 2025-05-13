import numpy as np
import pandas as pd
import smtplib
import os
import time as tm
from datetime import datetime
from configparser import ConfigParser
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') #non-interactive backends for png files
from matplotlib.ticker import MaxNLocator
import torch

class Util:
    def __init__(self, model_descr, dataset_type='notebooks', version=0, prefix=''):
        current_time = datetime.now()
        self.model_descr = model_descr
        self.start_time = current_time.strftime('%d/%m/%Y %H:%M:%S')
        self.start_time_timestamp = tm.time()
        self.version = str(version)
        prefix = prefix.lower() + '_' if prefix.strip() else ''
        self.base_filename =  prefix + self.version + '_' + current_time.strftime('%Y%m%d-%H%M%S')
        self.project_dir = str(Path(__file__).absolute().parent.parent)
        self.output_dir = os.path.join(self.project_dir, 'output', dataset_type)
        
    def plot(self, data, columns_name, x_label, y_label, title, enable=True, inline=False):
        if (enable):
            df = pd.DataFrame(data).T    
            df.columns = columns_name
            df.index += 1
            plot = df.plot(linewidth=2, figsize=(15,8), color=['darkgreen', 'orange'], grid=True);
            train = columns_name[0]
            val = columns_name[1]
            # find position of lowest validation loss
            idx_min_loss = df[val].idxmin()
            plot.axvline(idx_min_loss, linestyle='--', color='r',label='Best epoch');
            plot.legend();
            plot.set_xlim(0, len(df.index)+1);
            plot.xaxis.set_major_locator(MaxNLocator(integer=True))
            plot.set_xlabel(x_label, fontsize=12);
            plot.set_ylabel(y_label, fontsize=12);
            plot.set_title(title, fontsize=16);
            if (not inline):
                plot_dir = self.__create_dir('plots')
                filename = os.path.join(plot_dir, self.base_filename + '.png')
                plot.figure.savefig(filename, bbox_inches='tight');
        
    def send_email(self, model_info, enable=True):
        if (enable):
            config = ConfigParser()
            config.read(os.path.join(self.project_dir, 'config/mail_config.ini'))
            server = config.get('mailer','server')
            port = config.get('mailer','port')
            login = config.get('mailer','login')
            password = config.get('mailer', 'password')
            to = config.get('mailer', 'receiver')

            subject = 'Experiment execution [' + self.model_descr + ']'
            text = 'This is an email message to inform you that the python script has completed.'
            message = text + '\n' + str(self.get_time_info()) + '\n' + str(model_info)

            smtp = smtplib.SMTP_SSL(server, port)
            smtp.login(login, password)

            body = '\r\n'.join(['To: %s' % to,
                                'From: %s' % login,
                                'Subject: %s' % subject,
                                '', message])
            try:
                smtp.sendmail(login, [to], body)
                print ('email sent')
            except Exception:
                print ('error sending the email')

            smtp.quit()
    
    def save_loss(self, train_losses, val_losses, enable=True):
        if (enable):
            losses_dir = self.__create_dir('losses')
            train_dir, val_dir = self.__create_train_val_dir_in(losses_dir)
            train_filename = os.path.join(train_dir, self.base_filename + '.txt')
            val_filename = os.path.join(val_dir, self.base_filename + '.txt')
            np.savetxt(train_filename, train_losses, delimiter=",", fmt='%g')
            np.savetxt(val_filename, val_losses, delimiter=",", fmt='%g')
            
    def save_examples(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        output: torch.Tensor,
        step: int,
        iteration_number: int,
    ):
        print("Saving examples in grid_figures folder")
        features_tuple = {
            "tp": "Total precipitation",
            "r200": "Relative humidity at 200 hPa",
            "r700": "Relative humidity at 700 hPa",
            "r1000": "Relative humidity at 1000 hPa",
            "t200": "Temperature at 200 hPa",
            "t700": "Temperature at 700 hPa",
            "t1000": "Temperature at 1000 hPa",
            "u200": "U component of wind at 200 hPa",
            "u700": "U component of wind at 700 hPa",
            "u1000": "U component of wind at 1000 hPa",
            "v200": "V component of wind at 200 hPa",
            "v700": "V component of wind at 700 hPa",
            "v1000": "V component of wind at 1000 hPa",
            "speed200": "Speed of wind at 200 hPa",
            "speed700": "Speed of wind at 700 hPa",
            "speed1000": "Speed of wind at 1000 hPa",
            "w200": "Vertical velocity at 200 hPa",
            "w700": "Vertical velocity at 700 hPa",
            "w1000": "Vertical velocity at 1000 hPa",
        }

        run_name = os.getenv("run_name")

        os.makedirs(f"./grid_figures_{run_name}/iteration_{iteration_number}", exist_ok=True)
        
        # num_rows = len(features_tuple)
        # num_cols = 4
        # fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))

        cmap = 'YlGnBu' if self.base_filename.startswith('chirps') else 'viridis'

        def plot_single_axis(tensor, ax, title):
            im = ax.imshow(tensor, aspect="auto", cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
            return im

        sample = 0
        channel = 0

        seq_len = inputs.shape[2]
        sample = 0
        inputs = inputs[sample, :, :, :].cpu().numpy()
        output = output[sample, :, :, :].cpu().numpy()
        target = target[sample, :, :, :].cpu().numpy()

        print(f"inputs.shape: {inputs.shape}")
        print(f"output.shape: {output.shape}")
        print(f"target.shape: {target.shape}")
        for channel, (key, feature_name) in enumerate(features_tuple.items()):
            fig, axes = plt.subplots(3, 5, figsize=(12, 8))

            target_channel = channel
            if target.shape[0] == 1:
                target_channel = 0

            for t in range(seq_len):
                plot_single_axis(inputs[channel, t, :, :], axes[0, t], f"T{t}")
                plot_single_axis(output[target_channel, t, :, :], axes[1, t], f"T{t}")
                plot_single_axis(target[target_channel, t, :, :], axes[2, t], f"T{t}")

            row_labels = ["Input", "Prediction", "Target"]
            for row, label in enumerate(row_labels):
                axes[row, 0].text(
                    -0.1, 0.5, label,
                    va="center",
                    ha="right",
                    fontsize=8,
                    transform=axes[row, 0].transAxes
                )

            plt.suptitle(f"feature_name (sample={sample}, channel={channel})", fontsize=16)
            plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])
            plt.savefig(f"./grid_figures_{run_name}/iteration_{iteration_number}/{key}_figure.png", dpi=600)
            plt.close(fig)
            # break
        
        num_rows = len(features_tuple)
        num_cols = 4
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))

        timestep = inputs.shape[1] - 1
        print(f"shape inputs: {inputs.shape}")
        print(f"shape output: {output.shape}")
        print(f"shape target: {target.shape}")
        print(f"timestep: {timestep}")
        # return

        for channel, (key, feature_name) in enumerate(features_tuple.items()):
            axes[channel, 0].axis("off")
            axes[channel, 0].text(0.5, 0.5, feature_name, ha="center", va="center", fontsize=16)

            target_channel = channel
            if target.shape[0] == 1:
                target_channel = 0

            _ = axes[channel, 1].imshow(inputs[channel, timestep], aspect="auto", cmap="viridis")
            axes[channel, 1].set_title("Input")
            # fig.colorbar(im1, ax=axes[channel, 1], orientation="vertical")

            _ = axes[channel, 2].imshow(output[target_channel, timestep], aspect="auto", cmap="viridis")
            axes[channel, 2].set_title("Prediction")
            # fig.colorbar(im2, ax=axes[channel, 2], orientation="vertical")

            _ = axes[channel, 3].imshow(target[target_channel, timestep], aspect="auto", cmap="viridis")
            axes[channel, 3].set_title("Target")
            # fig.colorbar(im3, ax=axes[channel, 3], orientation="vertical")
            # break

        plt.tight_layout()
        plt.savefig(f"./grid_figures_{run_name}/iteration_{iteration_number}/full_grid_figure.png", dpi=300)
        plt.show()

    def get_checkpoint_filename(self):
        check_dir = self.__create_dir('checkpoints')
        filename = os.path.join(check_dir, self.base_filename + '.pth.tar')
        return filename
        
    def to_readable_time(self, timestamp):
        hours = int(timestamp / (60 * 60))
        minutes = int((timestamp % (60 * 60)) / 60)
        seconds = timestamp % 60.
        return f'{hours}:{minutes:>02}:{seconds:>05.2f}'
        
    def get_time_info(self):
        end_time = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        end_time_timestamp = tm.time()
        elapsed_time = end_time_timestamp - self.start_time_timestamp
        elapsed_time = self.to_readable_time(elapsed_time)
        time_info = {'model': self.model_descr,
                      'version': self.version,
                      'start_time': self.start_time,
                      'end_time': end_time,
                      'elapsed_time': elapsed_time}
        return time_info
       
    def get_mask_land(self):
        """
        Original chirps dataset has no ocean data, 
        so this mask is required to ensure that only land data is considered
        """
        filename = os.path.join(self.project_dir, 'data', 'chirps_mask_land.npy')
        mask_land = np.load(filename)
        mask_land = torch.from_numpy(mask_land).float()
        return mask_land
        
    @staticmethod         
    def generate_list_from(integer, size=3):
        if isinstance(integer,int):
            return [integer] * size
        return integer        
        
    def __create_train_val_dir_in(self, dir_path):
        train_dir = os.path.join(dir_path, 'train')
        os.makedirs(train_dir, exist_ok=True)
        val_dir = os.path.join(dir_path, 'val')   
        os.makedirs(val_dir, exist_ok=True)     
        return train_dir, val_dir
    
    def __create_dir(self, dir_name):
        new_dir = os.path.join(self.output_dir, dir_name, self.model_descr)
        os.makedirs(new_dir, exist_ok=True)
        return new_dir
        
    def __create_image_plot(self, tensor, ax, i, j, index, step, ax_input=False):
        cmap = 'YlGnBu' if self.base_filename.startswith('chirps') else 'viridis'
        tensor_numpy = tensor[0,:,index,:,:].squeeze().cpu().numpy()
        if step == 5 or ax_input:
            ax[j].imshow(np.flipud(tensor_numpy), cmap=cmap)
            ax[j].get_xaxis().set_visible(False)
            ax[j].get_yaxis().set_visible(False)
        else:
            ax[i][j].imshow(np.flipud(tensor_numpy), cmap=cmap)
            ax[i][j].get_xaxis().set_visible(False)
            ax[i][j].get_yaxis().set_visible(False)        
        return ax
        
    def __save_image_plot(self, figure, folder, name, step, fig_input=False):
        y = 0.7 if (step == 5 or fig_input) else 0.9
        figure.suptitle(name, y=y)
        filename = os.path.join(folder, name + '_' + self.base_filename + '.png') 
        figure.savefig(filename, dpi=300)