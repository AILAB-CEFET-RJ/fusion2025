import argparse
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from matplotlib.ticker import MaxNLocator

# Keep this import to reconstruct the test loader
from evaluate_training.NetCDFDataset import NetCDFDataset  # noqa: F401
from STConvS2S.model.stconvs2s import STConvS2S_R

seq_len = 5
input_timesteps = ["14:00", "15:00", "16:00", "17:00", "18:00"]
output_timesteps = ["19:00", "20:00", "21:00", "22:00", "23:00"]


MODEL_FILE = "/home/felipe/fusion2025/trained_models/ERA5+SIA-model.pth.tar-OK"
DATASET = "/home/felipe/rafaela-model/rafaela_model/models_evaluation/output_dataset_replaced-ERA5+SIA.nc"
test_split = 0.2
validation_split = 0.2
ds = xr.open_mfdataset(DATASET)


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")


def load_model():
    config = {
        "num_layers": 3,
        "hidden_dim": 32,
        "kernel_size": 5,
        "dropout_rate": 0.0,
        "step": 5,
        "device": device,
    }

    model = STConvS2S_R(
        (1, 19, 5, 9, 11),  # sample x channels x seq_len x height x width
        config["num_layers"],
        config["hidden_dim"],
        config["kernel_size"],
        config["device"],
        config["dropout_rate"],
        int(config["step"]),
    )

    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model


longitudes = np.array(
    [-45.053, -44.8029, -44.5528, -44.3027, -44.0526, -43.8025, -43.5524, -43.3023, -43.0522, -42.8021, -42.552]
)
latitudes = np.array([-21.801, -22.051, -22.301, -22.551, -22.801, -23.051, -23.301, -23.551, -23.802])


def plot_single_axis(
    tensor,
    ax,
    title,
    should_add_colorbar=False,
    scale_min=None,
    scale_max=None,
    is_map=True,
    longitudes=longitudes,
    latitudes=latitudes,
    cmap="coolwarm",
):
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    if scale_min is None or scale_max is None:
        vmin = tensor.min() if tensor.min() >= 0 else 0
        vmax = tensor.max()
    else:
        vmin = scale_min
        vmax = scale_max

    if is_map and longitudes is None or latitudes is None:
        raise ValueError("For map plotting, 'longitudes' and 'latitudes' must be provided.")

    im = ax.pcolormesh(
        longitudes,
        latitudes,
        tensor,
        cmap=cmap,
        alpha=0.75,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
    )

    ax.set_title(title, fontsize=12)
    ax.axis("off")

    if should_add_colorbar:
        fig = ax.get_figure()
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            fraction=0.046,
            pad=0.04,
            format="%g",
        )

        cbar.locator = MaxNLocator(nbins=3)
        cbar.update_ticks()

        cbar.ax.tick_params(labelsize=14)
    return im


def save_examples(
    inputs: torch.Tensor,
    target: torch.Tensor,
    output: torch.Tensor,
    step: int,
    iteration_number: int,
):
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

    sample = 0
    channel = 0

    seq_len = inputs.shape[2]
    sample = 0
    inputs = inputs[sample, :, :, :].cpu().numpy()
    output = output[sample, :, :, :].cpu().numpy()
    target = target[sample, :, :, :].cpu().numpy()

    for channel, (key, feature_name) in enumerate(list(features_tuple.items())[:1]):
        fig, axes = plt.subplots(3, 5, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})

        target_channel = channel
        if target.shape[0] == 1:
            target_channel = 0

        for t in range(seq_len):
            should_add_colorbar = True

            plot_single_axis(inputs[channel, t, :, :], axes[0, t], input_timesteps[t], should_add_colorbar, False)

            plot_single_axis(
                output[target_channel, t, :, :],
                axes[1, t],
                output_timesteps[t],
                should_add_colorbar=should_add_colorbar,
            )

            plot_single_axis(
                target[target_channel, t, :, :],
                axes[2, t],
                output_timesteps[t],
                should_add_colorbar,
            )

        row_labels = ["Input", "Prediction", "Target"]
        for row, label in enumerate(row_labels):
            axes[row, 0].text(
                -0.1,
                0.5,
                label,
                va="center",
                ha="right",
                fontsize=12,
                transform=axes[row, 0].transAxes,
            )

        plt.suptitle("Total precipitation prediction - 2024-12-20", fontsize=16)
        plt.tight_layout(rect=[0.05, 0.03, 1, 0.95])

        if not os.path.exists(f"./inference_grid_figures/iteration_{iteration_number}"):
            os.makedirs(f"./inference_grid_figures/iteration_{iteration_number}")

        plt.savefig(f"./inference_grid_figures/iteration_{iteration_number}/{key}_figure.png", dpi=600)
        print(f"Saved figure to ./inference_grid_figures/iteration_{iteration_number}/{key}_figure.png")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the model inference")
    parser.add_argument(
        "--dataset",
        type=str,
        help="The GFS+A to evaluate",
    )

    args = parser.parse_args()

    print(f"Loading inference dataset from {args.dataset}")

    test_loader = torch.load(args.dataset, weights_only=False)

    model = load_model()

    use_log = True

    model.eval()
    for batch_i, (inputs, target) in enumerate(test_loader):
        inputs, target = inputs.to(device), target.to(device)

        if use_log:
            precipitation_x = inputs[:, 0, :, :, :]
            inputs[:, 0, :, :, :] = torch.log1p(precipitation_x)

        with torch.no_grad():
            output = model(inputs)

        if use_log:
            output = torch.expm1(output)

        print("max output: ", output.max())
        output[output < 0] = 0
        print("min output: ", output.min())

        save_examples(inputs, target, output, 5, 0)

        level_50_inf = {
            "true": 0,
            "pred": {
                "0-5": 0,
                "5-25": 0,
                "25-50": 0,
                "50-inf": 0,
            },
        }

        level_25_50 = {
            "true": 0,
            "pred": {
                "0-5": 0,
                "5-25": 0,
                "25-50": 0,
                "50-inf": 0,
            },
        }

        level_5_25 = {
            "true": 0,
            "pred": {
                "0-5": 0,
                "5-25": 0,
                "25-50": 0,
                "50-inf": 0,
            },
        }

        level_0_5 = {
            "true": 0,
            "pred": {
                "0-5": 0,
                "5-25": 0,
                "25-50": 0,
                "50-inf": 0,
            },
        }

        # These are cells with dense stations in the spatiotemporal grid
        LATS_INDEXES = slice(4, 5)
        LONS_INDEXES = slice(5, 8)
        y_channel_0 = target[:, 0, :, LATS_INDEXES, LONS_INDEXES]
        y_pred_channel_0 = output[:, 0, :, LATS_INDEXES, LONS_INDEXES]

        mask_0_5 = y_channel_0 < 5
        mask_5_25 = (y_channel_0 >= 5) & (y_channel_0 < 25)
        mask_25_50 = (y_channel_0 >= 25) & (y_channel_0 < 50)
        mask_50_inf = y_channel_0 >= 50

        level_50_inf["true"] += mask_50_inf.sum().item()
        level_25_50["true"] += mask_25_50.sum().item()
        level_5_25["true"] += mask_5_25.sum().item()
        level_0_5["true"] += mask_0_5.sum().item()

        level_50_inf["pred"]["0-5"] += (mask_50_inf & (y_pred_channel_0 < 5)).sum().item()
        level_50_inf["pred"]["5-25"] += ((y_pred_channel_0 >= 5) & (y_pred_channel_0 < 25) & mask_50_inf).sum().item()
        level_50_inf["pred"]["25-50"] += ((y_pred_channel_0 >= 25) & (y_pred_channel_0 < 50) & mask_50_inf).sum().item()
        level_50_inf["pred"]["50-inf"] += ((y_pred_channel_0 >= 50) & mask_50_inf).sum().item()

        level_25_50["pred"]["0-5"] += ((y_pred_channel_0 < 5) & mask_25_50).sum().item()
        level_25_50["pred"]["5-25"] += ((y_pred_channel_0 >= 5) & (y_pred_channel_0 < 25) & mask_25_50).sum().item()
        level_25_50["pred"]["25-50"] += ((y_pred_channel_0 >= 25) & (y_pred_channel_0 < 50) & mask_25_50).sum().item()
        level_25_50["pred"]["50-inf"] += ((y_pred_channel_0 >= 50) & mask_25_50).sum().item()

        level_5_25["pred"]["0-5"] += ((y_pred_channel_0 < 5) & mask_5_25).sum().item()
        level_5_25["pred"]["5-25"] += ((y_pred_channel_0 >= 5) & (y_pred_channel_0 < 25) & mask_5_25).sum().item()
        level_5_25["pred"]["25-50"] += ((y_pred_channel_0 >= 25) & (y_pred_channel_0 < 50) & mask_5_25).sum().item()
        level_5_25["pred"]["50-inf"] += ((y_pred_channel_0 >= 50) & mask_5_25).sum().item()

        level_0_5["pred"]["0-5"] += ((y_pred_channel_0 < 5) & mask_0_5).sum().item()
        level_0_5["pred"]["5-25"] += ((y_pred_channel_0 >= 5) & (y_pred_channel_0 < 25) & mask_0_5).sum().item()
        level_0_5["pred"]["25-50"] += ((y_pred_channel_0 >= 25) & (y_pred_channel_0 < 50) & mask_0_5).sum().item()
        level_0_5["pred"]["50-inf"] += ((y_pred_channel_0 >= 50) & mask_0_5).sum().item()

        datasets_conf = {
            "0-5": level_0_5["pred"],
            "5-25": level_5_25["pred"],
            "25-50": level_25_50["pred"],
            "50-inf": level_50_inf["pred"],
        }

        confusion_matrix = pd.DataFrame(
            {
                pred_category: [
                    datasets_conf[true_category].get(pred_category, 0) for true_category in datasets_conf.keys()
                ]
                for pred_category in datasets_conf.keys()
            },
            index=datasets_conf.keys(),
        )

        latex_table = confusion_matrix.to_latex(
            header=True,
            index=True,
            index_names=False,
            column_format="lrrrr",
            bold_rows=False,
            multicolumn=False,
            multicolumn_format="c",
            multirow=False,
            float_format="%.4f",
        )

        print("Confusion matrix:")
        print(latex_table)

        break
