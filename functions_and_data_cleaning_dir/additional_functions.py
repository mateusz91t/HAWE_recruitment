import pandas as pd


def prepare_data_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Changes types in the data frame and prepares to work with.
    """
    data_frame["v_1"] = data_frame["v_1"].astype("category")
    data_frame["Ta4"] = data_frame["Ta4"].astype("str")
    data_frame["Ta4"] = data_frame["Ta4"].apply(lambda x: x.replace(",", "."))
    data_frame["Ta4"] = data_frame["Ta4"].astype("float32")
    data_frame["Timestamp"] = pd.to_datetime(data_frame["Timestamp"])
    data_frame.set_index(data_frame["Timestamp"], inplace=True)
    del data_frame["Timestamp"]
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


def draw_figure_axe_by_feature(df: pd.DataFrame, idxs: pd.Index):
    figure_columns = 2
    figure_rows = ceil(df.columns.size / figure_columns)
    fig, ax = plt.subplots(figure_rows, figure_columns, figsize=(16, 16))
    figure_axes = fig.get_axes()
    for plot_position in range(df.columns.size):
        column_name = df.columns[plot_position]
        plt.subplot(figure_rows, figure_columns, plot_position + 1)
        plt.plot(df.loc[idxs, column_name], marker=".")
        plt.title(column_name, y=0.1)
        figure_axes[plot_position].fill_between(
            df.loc[idxs, column_name].index,
            df.loc[idxs, column_name].min(),
            df.loc[idxs, column_name].max(),
            where=df.loc[idxs, column_name].isnull(),
            color="black",
            alpha=1,
            transform=figure_axes[plot_position].get_xaxis_transform(),
        )


def get_surrondings_around_indexes(idxs: pd.Index, around_minutes: int = 10):
    before_min_index = pd.date_range(
        idxs[0] - pd.Timedelta(minutes=10),
        idxs[0] - pd.Timedelta(minutes=1),
        freq="min",
    )
    after_max_index = pd.date_range(
        idxs[-1] + pd.Timedelta(minutes=1),
        idxs[-1] + pd.Timedelta(minutes=10),
        freq="min",
    )
    return [
        *before_min_index,
        *idxs,
        *after_max_index,
    ]
