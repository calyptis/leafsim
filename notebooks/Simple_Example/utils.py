"""
Helper functions for the simple example notebook.

Only used by the notebook and nowhere in the Python library.
"""
import os
import sys
from typing import Dict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils_common import conditional_background_gradient  # noqa: E402


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
