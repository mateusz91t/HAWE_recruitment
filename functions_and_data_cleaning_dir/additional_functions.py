import pandas as pd


def prepare_data_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Changes types in the data frame and prepares to work with.
    """
    data_frame["v_1"] = data_frame["v_1"].astype("category")
    data_frame["Ta4"] = data_frame["Ta4"].astype("str")
    data_frame["Ta4"] = data_frame["Ta4"].apply(lambda x: x.replace(",", "."))
    data_frame["Ta4"] = data_frame["Ta4"].astype("float32")
    return data_frame


def moving_mean(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Mades a dataframe with moving average base on a given window.

    Returns:
        DataFrame: returned columns are changed to orginalName__window__mean
    """
    df_rooling_mean = df.rolling(window=window).mean()
    df_rooling_mean.columns = [
        str(x) + f"__{window}__mean" for x in df_rooling_mean.columns
    ]

    return df_rooling_mean
