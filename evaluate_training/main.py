import argparse
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from STConvS2S.model.stconvs2s import STConvS2S_R

from .NetCDFDataset import NetCDFDataset
from .plot_distribution import categories, plot_bar, plot_histogram


def clean_precipitation_data(data, threshold=0.01, extreme_threshold=150.0, verbose=True):
    extreme_mask = data[:, :, :, :, 0] > extreme_threshold
    total_extremes = extreme_mask.sum()
    max_extreme = 0
    if total_extremes > 0:
        max_extreme = data[:, :, :, :, 0][extreme_mask].max()
    data[:, :, :, :, 0][extreme_mask] = 0

    total_changes_middle = 0
    max_changed_value_middle = 0

    for t in range(1, data.shape[1] - 1):
        mask = (data[:, t - 1, :, :, 0] == 0) & (data[:, t + 1, :, :, 0] == 0) & (data[:, t, :, :, 0] > threshold)
        changes_at_t = mask.sum()
        total_changes_middle += changes_at_t
        if changes_at_t > 0:
            max_val_at_t = data[:, t, :, :, 0][mask].max()
            max_changed_value_middle = max(max_changed_value_middle, max_val_at_t)
        data[:, t, :, :, 0][mask] = 0

    if verbose:
        print("=== Extreme Precipitation Removal ===")
        print(f"Total extreme values (>{extreme_threshold} mm/h) removed: {total_extremes}")
        print(f"Percentage of data removed: {100 * total_extremes / data.size:.6f}%")
        print(f"Maximum extreme value: {max_extreme}")

        print("\n=== Spurious Precipitation Removal ===")
        print(f"Total spurious values removed: {total_changes_middle}")
        print(f"Percentage of data changed: {100 * total_changes_middle / data.size:.4f}%")
        print(f"Maximum value changed: {max_changed_value_middle}")

    return data


def extract_filename_from_dataset_path(dataset: str):
    """
    Extract the filename from the dataset path.
    """
    if dataset.startswith("./"):
        dataset = dataset[2:]
    if dataset.endswith(".nc"):
        dataset = dataset[:-3]
    return dataset.split("/")[-1]


if __name__ == "__main__":
    # python -m evaluate_training.main --dataset "/home/user/ERA5+SIA.nc" --model "/home/user/fusion2025/trained_models/ERA5+SIA-model.pth.tar"
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to evaluate",
    )
    parser.add_argument(
        "--lead_time",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Lead time to evaluate",
        default=1,
    )

    args = parser.parse_args()

    print(f"""
        Running evaluation with args:
        dataset: {args.dataset}
        model: {args.model}
        lead_time: {args.lead_time}
    """)

    lead_times = {
        "T+1": 0,
        "T+2": 1,
        "T+3": 2,
        "T+4": 3,
        "T+5": 4,
    }

    LEAD_TIME_KEY = f"T+{args.lead_time}"
    LEAD_TIME = lead_times[LEAD_TIME_KEY]

    MODEL_FILE = args.model
    DATASET = args.dataset

    # These are cells with dense stations in the spatiotemporal grid
    LATS_INDEXES = slice(4, 5)
    LONS_INDEXES = slice(5, 8)

    KEY_NAME = os.path.basename(args.dataset)
    MODEL_DATASET_PATH = f"./models_evaluation/{KEY_NAME}/"
    Path(MODEL_DATASET_PATH).mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(DATASET)

    clean_precipitation_data(ds.x.values, verbose=True)
    print(f"ds.x.values[:, :, :, :, 0].max(): {ds.x.values[:, :, :, :, 0].max()}")
    print(np.log1p(ds.x[:, :, :, :, 0].max().values))
    print(np.log1p(ds.y[:, :, :, :, 0].max().values))
    clean_precipitation_data(ds.y.values, verbose=True)
    print(np.log1p(ds.y[:, :, :, :, 0].max().values))
    print(ds.y.values[:, :, :, :, 0].max())

    USE_MINMAX = False
    test_split = 0.2
    validation_split = 0.2

    if USE_MINMAX:
        train_dataset = NetCDFDataset(ds, test_split=test_split, validation_split=validation_split)
    test_dataset = NetCDFDataset(ds, test_split=test_split, validation_split=validation_split, is_test=True)
    print(f"Use MinMaxScaler: {USE_MINMAX}")
    if USE_MINMAX:
        for channel_idx in range(1, train_dataset.X.shape[1]):
            train_data = train_dataset.X[:, channel_idx].detach().cpu().numpy()
            original_shape = train_data.shape
            reshaped = train_data.reshape(-1, 1)

            scaler = MinMaxScaler().fit(reshaped)
            scaled_train = scaler.transform(reshaped).reshape(original_shape)

            train_dataset.X[:, channel_idx] = torch.tensor(
                scaled_train, dtype=train_dataset.X.dtype, device=train_dataset.X.device
            )

            test_data = test_dataset.X[:, channel_idx].detach().cpu().numpy()
            reshaped_test = test_data.reshape(-1, 1)
            scaled_test = scaler.transform(reshaped_test).reshape(test_data.shape)

            test_dataset.X[:, channel_idx] = torch.tensor(
                scaled_test, dtype=test_dataset.X.dtype, device=test_dataset.X.device
            )

            print(f"Scaled test data for channel index {channel_idx} using training scaler.")

    print(test_dataset.X.shape)
    print(test_dataset.y.shape)

    rain_dist = test_dataset.y[:, 0, :, LATS_INDEXES, LONS_INDEXES].numpy().flatten()
    fig, ax = plot_histogram(rain_dist)
    fig.savefig(f"{MODEL_DATASET_PATH}rain_distribution_histogram.png")
    print(f"Rain distribution histogram saved to {MODEL_DATASET_PATH}rain_distribution_histogram.png")
    fig, ax, rain_distribution = plot_bar(rain_dist)
    fig.savefig(f"{MODEL_DATASET_PATH}rain_distribution.png")
    print(f"Rain distribution bar plot saved to {MODEL_DATASET_PATH}rain_distribution.png")

    test_loader = DataLoader(dataset=test_dataset, shuffle=False, **{"batch_size": 15, "num_workers": 4})

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    config = {
        "num_layers": 3,
        "hidden_dim": 32,
        "kernel_size": 5,
        "dropout_rate": 0.0,
        "step": 5,
        "device": device,
    }

    model = STConvS2S_R(
        test_dataset.X.shape,
        config["num_layers"],
        config["hidden_dim"],
        config["kernel_size"],
        config["device"],
        config["dropout_rate"],
        int(config["step"]),
    )

    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    level_50_inf = {
        "true": 0,
        "pred": {
            "0-5": 0,
            "5-25": 0,
            "25-50": 0,
            "50-inf": 0,  # in a perfect scenario, this should be equal to the true value
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

    def process_batches():
        less_than_0 = 0

        levels_squared_errors = {
            "0-5": np.nan,
            "5-25": np.nan,
            "25-50": np.nan,
            "50-inf": np.nan,
        }
        levels_counts = {
            "0-5": 0,
            "5-25": 0,
            "25-50": 0,
            "50-inf": 0,
        }
        levels_errors_sum = {
            "0-5": np.nan,
            "5-25": np.nan,
            "25-50": np.nan,
            "50-inf": np.nan,
        }

        levels_abs = {
            "0-5": [],
            "5-25": [],
            "25-50": [],
            "50-inf": [],
        }
        levels_bias = {
            "0-5": [],
            "5-25": [],
            "25-50": [],
            "50-inf": [],
        }

        rain_distribution_true = {
            "light": 0,
            "moderate": 0,
            "heavy": 0,
            "extreme": 0,
        }
        rain_distribution_pred = {
            "light": 0,
            "moderate": 0,
            "heavy": 0,
            "extreme": 0,
        }

        use_log = True

        model.eval()
        for i, (X, y) in enumerate(tqdm(test_loader, total=len(test_loader))):
            X = X.to(device)
            y = y.to(device)

            if use_log:
                precipitation_x = X[:, 0, :, :, :]
                X[:, 0, :, :, :] = torch.log1p(precipitation_x)

            with torch.no_grad():
                y_pred = model(X)

            if use_log:
                y_pred = np.expm1(y_pred)

            y_channel_0 = y[:, 0, :, LATS_INDEXES, LONS_INDEXES]
            y_pred_channel_0 = y_pred[:, 0, :, LATS_INDEXES, LONS_INDEXES]

            y_channel_0 = y_channel_0[:, slice(LEAD_TIME, LEAD_TIME + 1), :, :]
            y_pred_channel_0 = y_pred_channel_0[:, slice(LEAD_TIME, LEAD_TIME + 1), :, :]

            y_pred_channel_0[y_pred_channel_0 < 0] = 0
            # LeakyRelu may output negative values, in this case, we set them to 0 (no rain)

            abs_diff = torch.abs(y_pred_channel_0 - y_channel_0)

            mask_0_5 = y_channel_0 < 5
            if mask_0_5.any():
                category_abs = torch.sqrt(torch.mean(abs_diff[mask_0_5]))
                levels_abs["0-5"].append(category_abs)
                category_bias = torch.mean(y_pred_channel_0[mask_0_5] - y_channel_0[mask_0_5])
                levels_bias["0-5"].append(category_bias)

                count = mask_0_5.sum().item()
                levels_counts["0-5"] += count
                if levels_squared_errors["0-5"] is np.nan:
                    levels_squared_errors["0-5"] = 0.0
                levels_squared_errors["0-5"] += abs_diff[mask_0_5].sum().item()
                if levels_errors_sum["0-5"] is np.nan:
                    levels_errors_sum["0-5"] = 0.0
                levels_errors_sum["0-5"] += (y_pred_channel_0[mask_0_5] - y_channel_0[mask_0_5]).sum().item()

            mask_5_25 = (y_channel_0 >= 5) & (y_channel_0 < 25)
            if mask_5_25.any():
                category_abs = torch.sqrt(torch.mean(abs_diff[mask_5_25]))
                levels_abs["5-25"].append(category_abs)
                category_bias = torch.mean(y_pred_channel_0[mask_5_25] - y_channel_0[mask_5_25])
                levels_bias["5-25"].append(category_bias)

                count = mask_5_25.sum().item()
                levels_counts["5-25"] += count
                if levels_squared_errors["5-25"] is np.nan:
                    levels_squared_errors["5-25"] = 0.0
                levels_squared_errors["5-25"] += abs_diff[mask_5_25].sum().item()
                if levels_errors_sum["5-25"] is np.nan:
                    levels_errors_sum["5-25"] = 0.0
                levels_errors_sum["5-25"] += (y_pred_channel_0[mask_5_25] - y_channel_0[mask_5_25]).sum().item()

            mask_25_50 = (y_channel_0 >= 25) & (y_channel_0 < 50)
            if mask_25_50.any():
                category_abs = torch.sqrt(torch.mean(abs_diff[mask_25_50]))
                levels_abs["25-50"].append(category_abs)
                category_bias = torch.mean(y_pred_channel_0[mask_25_50] - y_channel_0[mask_25_50])
                levels_bias["25-50"].append(category_bias)

                count = mask_25_50.sum().item()
                levels_counts["25-50"] += count
                if levels_squared_errors["25-50"] is np.nan:
                    levels_squared_errors["25-50"] = 0.0
                levels_squared_errors["25-50"] += abs_diff[mask_25_50].sum().item()
                if levels_errors_sum["25-50"] is np.nan:
                    levels_errors_sum["25-50"] = 0.0
                levels_errors_sum["25-50"] += (y_pred_channel_0[mask_25_50] - y_channel_0[mask_25_50]).sum().item()

            mask_50_inf = y_channel_0 >= 50
            if mask_50_inf.any():
                category_abs = torch.sqrt(torch.mean(abs_diff[mask_50_inf]))
                levels_abs["50-inf"].append(category_abs)
                category_bias = torch.mean(y_pred_channel_0[mask_50_inf] - y_channel_0[mask_50_inf])
                levels_bias["50-inf"].append(category_bias)

                count = mask_50_inf.sum().item()
                levels_counts["50-inf"] += count
                if levels_squared_errors["50-inf"] is np.nan:
                    levels_squared_errors["50-inf"] = 0.0
                levels_squared_errors["50-inf"] += abs_diff[mask_50_inf].sum().item()
                if levels_errors_sum["50-inf"] is np.nan:
                    levels_errors_sum["50-inf"] = 0.0
                levels_errors_sum["50-inf"] += (y_pred_channel_0[mask_50_inf] - y_channel_0[mask_50_inf]).sum().item()

            rain_distribution_true["light"] += mask_0_5.sum().item()
            rain_distribution_true["moderate"] += mask_5_25.sum().item()
            rain_distribution_true["heavy"] += mask_25_50.sum().item()
            rain_distribution_true["extreme"] += mask_50_inf.sum().item()

            level_50_inf["true"] += mask_50_inf.sum().item()
            level_25_50["true"] += mask_25_50.sum().item()
            level_5_25["true"] += mask_5_25.sum().item()
            level_0_5["true"] += mask_0_5.sum().item()

            level_50_inf["pred"]["0-5"] += (mask_50_inf & (y_pred_channel_0 < 5)).sum().item()
            level_50_inf["pred"]["5-25"] += (
                ((y_pred_channel_0 >= 5) & (y_pred_channel_0 < 25) & mask_50_inf).sum().item()
            )
            level_50_inf["pred"]["25-50"] += (
                ((y_pred_channel_0 >= 25) & (y_pred_channel_0 < 50) & mask_50_inf).sum().item()
            )
            level_50_inf["pred"]["50-inf"] += ((y_pred_channel_0 >= 50) & mask_50_inf).sum().item()

            level_25_50["pred"]["0-5"] += ((y_pred_channel_0 < 5) & mask_25_50).sum().item()
            level_25_50["pred"]["5-25"] += ((y_pred_channel_0 >= 5) & (y_pred_channel_0 < 25) & mask_25_50).sum().item()
            level_25_50["pred"]["25-50"] += (
                ((y_pred_channel_0 >= 25) & (y_pred_channel_0 < 50) & mask_25_50).sum().item()
            )
            level_25_50["pred"]["50-inf"] += ((y_pred_channel_0 >= 50) & mask_25_50).sum().item()

            level_5_25["pred"]["0-5"] += ((y_pred_channel_0 < 5) & mask_5_25).sum().item()
            level_5_25["pred"]["5-25"] += ((y_pred_channel_0 >= 5) & (y_pred_channel_0 < 25) & mask_5_25).sum().item()
            level_5_25["pred"]["25-50"] += ((y_pred_channel_0 >= 25) & (y_pred_channel_0 < 50) & mask_5_25).sum().item()
            level_5_25["pred"]["50-inf"] += ((y_pred_channel_0 >= 50) & mask_5_25).sum().item()

            level_0_5["pred"]["0-5"] += ((y_pred_channel_0 < 5) & mask_0_5).sum().item()
            level_0_5["pred"]["5-25"] += ((y_pred_channel_0 >= 5) & (y_pred_channel_0 < 25) & mask_0_5).sum().item()
            level_0_5["pred"]["25-50"] += ((y_pred_channel_0 >= 25) & (y_pred_channel_0 < 50) & mask_0_5).sum().item()
            level_0_5["pred"]["50-inf"] += ((y_pred_channel_0 >= 50) & mask_0_5).sum().item()

            rain_distribution_pred["light"] += (y_pred_channel_0 < 5).sum().item()
            rain_distribution_pred["moderate"] += ((y_pred_channel_0 >= 5) & (y_pred_channel_0 < 25)).sum().item()
            rain_distribution_pred["heavy"] += ((y_pred_channel_0 >= 25) & (y_pred_channel_0 < 50)).sum().item()
            rain_distribution_pred["extreme"] += (y_pred_channel_0 >= 50).sum().item()

            less_than_0 += (y_pred_channel_0 < 0).sum().item()

        levels_abs_all_batches = {}
        levels_bias_all_batches = {}

        for category in levels_counts.keys():
            if levels_counts[category] > 0:
                levels_abs_all_batches[category] = levels_squared_errors[category] / levels_counts[category]
                levels_bias_all_batches[category] = levels_errors_sum[category] / levels_counts[category]
            else:
                levels_abs_all_batches[category] = float("nan")
                levels_bias_all_batches[category] = float("nan")

        return (
            levels_abs,
            rain_distribution_true,
            rain_distribution_pred,
            less_than_0,
            levels_bias,
            levels_abs_all_batches,
            levels_bias_all_batches,
        )

    (
        levels_abs,
        rain_distribution_true,
        rain_distribution_pred,
        less_than_0,
        levels_bias,
        levels_abs_all_batches,
        levels_bias_all_batches,
    ) = process_batches()
    print("Rain Distribution (True):", rain_distribution_true)
    print("Rain Distribution (Pred):", rain_distribution_pred)
    print("Levels Abs Difference (All Batches):", levels_abs_all_batches)
    print("Levels Bias (All Batches):", levels_bias_all_batches)

    a = rain_distribution_true["light"] / sum(rain_distribution_true.values()) * 100
    b = rain_distribution_true["moderate"] / sum(rain_distribution_true.values()) * 100
    c = rain_distribution_true["heavy"] / sum(rain_distribution_true.values()) * 100
    d = rain_distribution_true["extreme"] / sum(rain_distribution_true.values()) * 100
    print("light percentage:", np.round(a, 2))
    print("moderate percentage:", np.round(b, 2))
    print("heavy percentage:", np.round(c, 2))
    print("extreme percentage:", np.round(d, 2))
    print("Total percentage:", a + b + c + d)

    assert rain_distribution_true["extreme"] == level_50_inf["true"]
    assert rain_distribution_true["heavy"] == level_25_50["true"]
    assert rain_distribution_true["moderate"] == level_5_25["true"]
    assert rain_distribution_true["light"] == level_0_5["true"]

    try:
        assert rain_distribution_true["extreme"] == level_50_inf["true"] == sum(level_50_inf["pred"].values())
    except AssertionError:
        print("Extreme assertion failed")
        print(f"True: {rain_distribution_true['extreme']}")
        print(f"Level 50-inf: {level_50_inf['true']}")
        print(f"Pred: {sum(level_50_inf['pred'].values())}\n")

    try:
        assert rain_distribution_true["heavy"] == level_25_50["true"] == sum(level_25_50["pred"].values())
    except AssertionError:
        print("Heavy assertion failed")
        print(f"True: {rain_distribution_true['heavy']}")
        print(f"Level 25-50: {level_25_50['true']}")
        print(f"Pred: {sum(level_25_50['pred'].values())}\n")

    try:
        assert rain_distribution_true["moderate"] == level_5_25["true"] == sum(level_5_25["pred"].values())
    except AssertionError:
        print("Moderate assertion failed")
        print(f"True: {rain_distribution_true['moderate']}")
        print(f"Level 5-25: {level_5_25['true']}")
        print(f"Pred: {sum(level_5_25['pred'].values())}\n")

    try:
        assert rain_distribution_true["light"] == level_0_5["true"] == sum(level_0_5["pred"].values())
    except AssertionError:
        print("Light assertion failed")
        print(f"True: {rain_distribution_true['light']}")
        print(f"Level 0-5: {level_0_5['true']}")
        print(f"Pred: {sum(level_0_5['pred'].values())}\n")

    categories = list(levels_abs.keys())
    errors1 = [list(levels_abs[cat]) for cat in categories]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(errors1, tick_labels=categories)
    plt.title("MAE per Category")
    plt.xlabel("Levels")
    plt.ylabel("MAE")
    plt.savefig(f"{MODEL_DATASET_PATH}mae_per_category.png")
    print(f"MAE per Category saved to {MODEL_DATASET_PATH}mae_per_category.png")

    bias1 = [list(levels_bias[cat]) for cat in categories]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(bias1, tick_labels=categories)
    plt.title("Bias per Category")
    plt.xlabel("Levels")
    plt.ylabel("Bias")
    plt.savefig(f"{MODEL_DATASET_PATH}bias_per_category.png")
    print(f"Bias per Category saved to {MODEL_DATASET_PATH}bias_per_category.png")

    abss = {}
    biases = {}
    for cat in categories:
        abss[cat] = levels_abs_all_batches[cat]

    for cat in categories:
        biases[cat] = levels_bias_all_batches[cat]

    with open(f"{MODEL_DATASET_PATH}mae.pkl", "wb") as f:
        pickle.dump(abss, f)

    with open(f"{MODEL_DATASET_PATH}bias.pkl", "wb") as f:
        pickle.dump(biases, f)

    print(f"MAE: {abss}")
    print(f"BIAS: {biases}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel("Precipitation (mm)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)

    bar_width = 0.4
    index = np.arange(len(rain_distribution_true))

    rects1 = ax.bar(index, rain_distribution_true.values(), bar_width, color="lightgreen", label="True", log=True)

    rects2 = ax.bar(
        index + bar_width,
        rain_distribution_pred.values(),
        bar_width,
        color="lightblue",
        label="Pred",
        log=True,
    )

    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(rain_distribution_true.keys())
    ax.legend(title="", fontsize=8, title_fontsize=9)
    plt.savefig(f"{MODEL_DATASET_PATH}rain_distribution_true_pred.png")
    print(f"Rain distribution true vs pred saved to {MODEL_DATASET_PATH}rain_distribution_true_pred.png")

    def plot_prediction(data: dict, title: str, y_text_offset=1, highlight_index=None):
        categories = list(data.keys())
        values = list(data.values())

        plt.figure(figsize=(8, 6))

        colors = ["skyblue"] * len(categories)
        if highlight_index is not None and 0 <= highlight_index < len(categories):
            colors[highlight_index] = "lightgreen"

        plt.bar(categories, values, color=colors, edgecolor="black")

        total = sum(values)
        plt.title(f"{title}", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        if total != 0:
            for i, value in enumerate(values):
                if value == 0:
                    continue
                plt.text(i, value + y_text_offset, f"{value} ({value / total:.2%})", ha="center", fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{MODEL_DATASET_PATH}{title}_confusion_matrix.png")
        print(f"Confusion matrix for {title} saved to {MODEL_DATASET_PATH}{title}_confusion_matrix.png")

    plot_prediction(level_50_inf["pred"], "[50-inf)", 0.4, 3)
    plot_prediction(level_25_50["pred"], "[25-50)", 100, 2)
    plot_prediction(level_5_25["pred"], "[5-25)", 100, 1)
    plot_prediction(level_0_5["pred"], "[0-5)", 500.5, 0)

    datasets = {
        "0-5": level_0_5["pred"],
        "5-25": level_5_25["pred"],
        "25-50": level_25_50["pred"],
        "50-inf": level_50_inf["pred"],
    }

    def get_confusion_matrix():
        confusion_matrix = pd.DataFrame(
            {
                pred_category: [datasets[true_category].get(pred_category, 0) for true_category in datasets.keys()]
                for pred_category in datasets.keys()
            },
            index=datasets.keys(),
        )

        confusion_matrix.index = [f"\\textnormal{{[{index})}}" for index in confusion_matrix.index]

        confusion_matrix.columns = [f"[{col})" for col in confusion_matrix.columns]

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

        with open(f"{MODEL_DATASET_PATH}confusion_matrix.tex", "w") as file:
            file.write(latex_table)

        confusion_matrix.index = datasets.keys()
        confusion_matrix.columns = datasets.keys()
        return confusion_matrix

    confusion_matrix = get_confusion_matrix()
    print(confusion_matrix)

    def get_ytrue_ypred(confusion_matrix: pd.DataFrame):
        y_true = []
        y_pred = []
        for true_label in confusion_matrix.index:
            for pred_label in confusion_matrix.columns:
                count = confusion_matrix.loc[true_label, pred_label]
                y_true.extend([true_label] * count)
                y_pred.extend([pred_label] * count)
        return y_true, y_pred

    y_true, y_pred = get_ytrue_ypred(confusion_matrix)

    sample_weight = compute_sample_weight("balanced", y_true)
    print(f"unique weights: {np.unique(sample_weight)}")

    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=confusion_matrix.index.to_list())
    f1_weighted = f1_score(
        y_true,
        y_pred,
        labels=confusion_matrix.index.to_list(),
        average="weighted",
        sample_weight=sample_weight,
    )
    f1_macro = f1_score(y_true, y_pred, labels=confusion_matrix.index.to_list(), average="macro")

    for label, f1 in zip(confusion_matrix.index.to_list(), per_class_f1):
        print(f"F1 Score for {label}: {f1:.4f}")

    per_class_f1_file = {}
    for i, label in enumerate(confusion_matrix.index.to_list()):
        if i < len(per_class_f1):
            per_class_f1_file[label] = per_class_f1[i]
        else:
            per_class_f1_file[label] = np.nan

    with open(f"{MODEL_DATASET_PATH}f1_per_class_{LEAD_TIME_KEY}.pkl", "wb") as f:
        pickle.dump(per_class_f1_file, f)
        print(f"f1_per_class_{LEAD_TIME_KEY}.pkl written")

    with open(f"{MODEL_DATASET_PATH}f1_score_weighted.pkl", "wb") as f:
        print(f"""
            Writting f1_score_weighted.pkl:
            {MODEL_DATASET_PATH}f1_score_weighted.pkl
            f1_weighted={f1_weighted}
        """)
        pickle.dump(f1_weighted, f)

    with open(f"{MODEL_DATASET_PATH}f1_score_macro.pkl", "wb") as f:
        print(f"""
            Writting f1_score_macro.pkl:
            {MODEL_DATASET_PATH}f1_score_macro.pkl
            f1_macro={f1_macro}
        """)
        pickle.dump(f1_macro, f)
