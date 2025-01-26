import random
import pandas as pd
import seaborn as sns
from typing import Tuple
import matplotlib.pyplot as plt


def plot_target(data: pd.DataFrame) -> None:
    """
    Plots the distribution of labels in the dataset.

    Args:
        data (pd.DataFrame): A DataFrame containing the 'labels' column to plot.
    """

    sns.set(style="whitegrid")
    sns.set_context("talk")
    plt.figure(figsize=(10, 8))

    sns.countplot(data=data, x="labels", palette="Blues_d", hue="labels", legend=False)

    plt.title("Distribution of Labels", fontsize=18, pad=20)
    plt.xlabel("Labels", fontsize=14)
    plt.ylabel("Count", fontsize=14)

    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    plt.show()


def random_image(
    data: pd.DataFrame,
    num_sample: int = 2,
    cmap: str = "gray",
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 100,
    title_fontsize: int = 14,
    title_fontweight: str = "bold",
) -> None:
    """
    Displays a random sample of images from the dataset.

    Args:
        data (pd.DataFrame): A DataFrame containing 'images' and 'labels' columns.
        num_sample (int): The number of images to display. Default is 2.
        cmap (str): The colormap to use for displaying images. Default is "gray".
        figsize (Tuple[int, int]): The size of the figure. Default is (8, 8).
        dpi (int): The resolution of the figure. Default is 100.
        title_fontsize (int): The font size of the title. Default is 14.
        title_fontweight (str): The font weight of the title. Default is "bold".
    """
    nrows = (num_sample + 1) // 2

    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=figsize, dpi=dpi)

    axes = axes.flatten()

    for i in range(num_sample):
        index = random.randint(0, len(data.index) - 1)

        image = data.iloc[index]["images"]
        label = data.iloc[index]["labels"]

        axes[i].imshow(image, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")

        axes[i].set_title(
            f"Sample {i + 1} (Label: {'Yes' if label else 'No'})",
            fontsize=title_fontsize,
            fontweight=title_fontweight,
            pad=10,
        )

        axes[i].axis("off")

    plt.tight_layout()
    plt.show()
