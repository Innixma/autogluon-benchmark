import numpy as np
import pandas as pd
import plotly.express as px

from ..evaluation.elo.elo_utils import (
    compute_pairwise_win_fraction,
    predict_win_rate,
)


def plot_winrate_expectation_by_elo(bootstrap_elo_lu):
    win_rate = predict_win_rate(dict(bootstrap_elo_lu.quantile(0.5)))
    ordered_models = win_rate.mean(axis=1).sort_values(ascending=False).index
    ordered_models = ordered_models[:30]
    fig = px.imshow(win_rate.loc[ordered_models, ordered_models],
                    color_continuous_scale='RdBu', text_auto=".2f",
                    title="Predicted Win Rate Using Elo Ratings for Model A in an A vs. B Battle")
    fig.update_layout(xaxis_title="Model B",
                      yaxis_title="Model A",
                      xaxis_side="top", height=900, width=900,
                      title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
                      "Model A: %{y}<br>Model B: %{x}<br>Win Rate: %{z}<extra></extra>")
    return fig


def visualize_pairwise_win_fraction(battles, title, max_num_models=30):
    row_beats_col = compute_pairwise_win_fraction(battles, max_num_models)
    fig = px.imshow(row_beats_col, color_continuous_scale='RdBu',
                    text_auto=".2f", title=title)
    fig.update_layout(xaxis_title=" Model B: Loser",
                  yaxis_title="Model A: Winner",
                  xaxis_side="top", height=900, width=900,
                  title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
                  "Model A: %{y}<br>Model B: %{x}<br>Fraction of A Wins: %{z}<extra></extra>")

    return fig


def visualize_bootstrap_scores(df: pd.DataFrame, title: str):
    bars = pd.DataFrame(dict(
        lower=df.quantile(.025),
        rating=df.quantile(.5),
        upper=df.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['rating_rounded'] = np.round(bars['rating'], 0).astype(int)
    num_models = len(bars["model"].unique())
    fig = px.scatter(bars, x="model", y="rating", error_y="error_y",
                     error_y_minus="error_y_minus", text="rating_rounded",
                     title=title)
    fig.update_layout(xaxis_title="Model", yaxis_title="Rating",
                      height=600, width=200 + num_models*40)
    return fig
