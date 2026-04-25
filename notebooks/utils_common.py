"""Shared notebook utilities."""
import matplotlib.pylab as plt
import pandas as pd
from matplotlib import colors


def conditional_background_gradient(
    _,
    df: pd.DataFrame,
    cmap: str = "PuBu",
    lower_threshold: float = 0,
    upper_threshold: float = 0.5,
    col: str = "diff_price",
) -> list[str]:
    """
    Return background colouring for a column conditioned on a different column.

    (from a potentially even different dataframe - if same size).

    :param _: Column to stylise (accepted by the function but never used)
    :param df: Dataframe containing the column on which the gradient is based
    :param cmap: Colourmap to use (accepts valid matplotlib cmaps)
    :param lower_threshold: Value at which the cmap starts
    :param upper_threshold: Value at which the cmap maxes out
    :param col: Column in df on which the background gradient is based
    """
    # Adapted from
    # https://stackoverflow.com/questions/47391948/pandas-style-background-gradient-using-other-dataframe
    a = df.loc[:, col].copy()
    norm = colors.Normalize(lower_threshold, upper_threshold)
    normed = norm(a.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ["background-color: %s" % color for color in c]
