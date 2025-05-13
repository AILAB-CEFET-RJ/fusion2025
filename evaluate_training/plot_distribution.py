import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

categories = {
    "light": (0, 5),
    "moderate": (5, 25),
    "heavy": (25, 50),
    "extreme": (50, "inf"),
}


def _get_bar_color(value):
    if 0 <= value < 5:
        return "lightblue"
    elif 5 <= value < 25:
        return "lightgreen"
    elif 25 <= value < 50:
        return "gold"
    elif value >= 50:
        return "tomato"
    return "lightgray"


def _get_rain_distribution(data_flatten: npt.NDArray[np.float64]) -> dict[str, int]:
    rain_distribution = {}
    for category, (lower, upper) in categories.items():
        if upper == "inf":
            mask = data_flatten >= lower
        else:
            mask = (data_flatten >= lower) & (data_flatten < upper)
        rain_distribution[category] = np.sum(mask)
    return rain_distribution


def _get_fig_ax():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel("Precipitation (mm/h)", fontsize=12)
    ax.legend(
        handles=[
            mpatches.Patch(color="lightblue", label="Weak [0, 5)"),
            mpatches.Patch(color="lightgreen", label="Moderate [5, 25)"),
            mpatches.Patch(color="gold", label="Heavy [25, 50)"),
            mpatches.Patch(color="tomato", label="Extreme [50, ∞)"),
        ],
        title="Precipitation Levels",
        fontsize=11,
        title_fontsize=11,
    )
    return fig, ax


def plot_histogram(data_flatten, log=True):
    fig, ax = _get_fig_ax()
    binwidth = 5
    bins = np.arange(0, data_flatten.max() + binwidth, binwidth)
    n, bins, patches = ax.hist(data_flatten, bins=bins, edgecolor="black", log=log)
    for patch, value in zip(patches, bins):
        patch.set_facecolor(_get_bar_color(value))
    return fig, ax


def plot_bar(data_flatten, log=True):
    rain_distribution = _get_rain_distribution(data_flatten)
    print(rain_distribution)
    fig, ax = _get_fig_ax()
    ax.bar(
        rain_distribution.keys(),
        rain_distribution.values(),
        color=["lightblue", "lightgreen", "gold", "tomato"],
        edgecolor="black",
        log=log,
    )
    return fig, ax, rain_distribution
