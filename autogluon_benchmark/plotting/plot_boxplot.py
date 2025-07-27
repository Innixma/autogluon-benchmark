import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_boxplot(
    data: pd.DataFrame,
    x: str,
    y: str = "framework",
    title: str = None,
    sort: bool = True,
    higher_is_better: bool = True,
    xlim: tuple = None,
    xscale: str = None,
    palette: str = "husl",
    save_path: str = None,
    show: bool = True,
):
    if sort:
        order = data[[y, x]].groupby([y]).agg(["median", "mean"]).sort_values(by=[(x, "median"), (x, "mean")], ascending=not higher_is_better)
        order = list(order.index)
    else:
        order = None

    fig, ax = plt.subplots(figsize=(20, 10))
    ax = sns.boxplot(
        data=data,
        ax=ax,
        y=y,
        x=x,
        order=order,
        hue=y,
        hue_order=order,
        palette=palette,
        legend=False,
    )
    if xlim is not None:
        ax.set(xlim=xlim)
    if title is not None:
        ax.set_title(title)
    if xscale is not None:
        ax.set_xscale(xscale)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)

    if show:
        plt.show()