"""
Helper functions for the notebooks.

Only used by them and nowhere in the Python library.
"""
from typing import Dict

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
    Return background colouring for a given column conditioned on a different column.

    (from a potentially even different dataframe - if same size).

    :param _: Column to stylise (needs to be accepted by the function but is never used)
    :param df: Dataframe containing the column based on which
               the background gradient should be based
    :param cmap: Colourmap to use for the background gradient
                 (accepts valid matplotlib cmaps)
    :param lower_threshold: The value at which the cmap should start
                            (first colour will be used for this value
                            and all that are smaller)
    :param upper_threshold: The value at which the cmap should max out
                            (last colour will be used of this value
                            and all that are larger)
    :param col: Column in df on which the background gradient should be based
    """
    # Adapted from
    # https://stackoverflow.com/questions/47391948/pandas-style-background-gradient-using-other-dataframe
    a = df.loc[:, col].copy()
    norm = colors.Normalize(lower_threshold, upper_threshold)
    normed = norm(a.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ["background-color: %s" % color for color in c]


# =========================
# For the simple example


def apply_formatting(
    to_explain: pd.DataFrame,
    df_explanation: pd.DataFrame,
    formatting: Dict[str, str],
) -> pd.DataFrame:
    """
    NOTE: Only works for the example shown in notebooks/Example.ipynb.

    Generates a stylised table of the
    most similar iris flowers for a chosen flower to be explained.

    Styling:
    - background gradient of numeric feature columns based on the percentage deviation
      from the car to be explained.

    :param df_explanation: Dataframe containing training examples
                           (from which the most similar cars are identified)
    :param to_explain: Dataframe containing test example
                       (observation we want to generate explanations for)
    :param formatting: Custom formatting of the feature columns passed on to
                       pd.DataFrame().style.format(...)
    :return: Stylised dataframe holding the top N most similar observation
    """
    d = to_explain.to_dict(orient="records")[0]

    cols_to_format = df_explanation.columns

    df_to_show = df_explanation.copy()
    for c in cols_to_format:
        if c == "similarity":
            df_to_show[c + "_diff"] = df_to_show[c]
        else:
            df_to_show[c + "_diff"] = (df_to_show[c] - d[c]).abs() / d[c]

    df_to_show = (
        df_to_show[cols_to_format]
        .style.format(formatting)
        # Add a background colour to the certain column based on how
        # far away they are from the datapoint to explain (in relative terms)
        .apply(
            conditional_background_gradient,
            subset="sepal length (cm)",
            cmap="Reds",
            df=df_to_show,
            col="sepal length (cm)_diff",
            upper_threshold=1,
        )
        .apply(
            conditional_background_gradient,
            subset="sepal width (cm)",
            cmap="Reds",
            df=df_to_show,
            col="sepal width (cm)_diff",
            upper_threshold=1,
        )
        .apply(
            conditional_background_gradient,
            subset="petal length (cm)",
            cmap="Reds",
            df=df_to_show,
            col="petal length (cm)_diff",
            upper_threshold=1,
        )
        .apply(
            conditional_background_gradient,
            subset="petal width (cm)",
            cmap="Reds",
            df=df_to_show,
            col="petal width (cm)_diff",
            upper_threshold=1,
        )
        .background_gradient(subset="similarity", cmap="Blues", vmax=1, vmin=0)
    )

    return df_to_show
