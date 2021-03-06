import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import missingno as msno

from functions_and_data_cleaning_dir.additional_functions import (
    prepare_data_frame,
    draw_figure_axe_by_feature,
    get_surrondings_around_indexes,
    one_hot_encoding,
)


# read data
df = pd.read_csv("df_task_final.csv")
# clean
df = prepare_data_frame(df)
# one hot encoding - change to continuous values
df = one_hot_encoding(df, "v_1", 50)

# each chart with nulls
draw_figure_axe_by_feature(df, df.index)

# all in 1 chart
fig, ax = plt.subplots(figsize=(15, 10))
plt.xticks(rotation=45)
for col in range(df.columns.size):
    ax.fill_between(
        df.iloc[:, col].index,
        df.iloc[:, col].min(),
        df.iloc[:, col].max(),
        where=df.iloc[:, col].isnull(),
        color="black",
        alpha=1,
        transform=ax.get_xaxis_transform(),
    )
    ticks = df.index[(df.index.hour == col) & (df.index.minute == col)]
    plt.xticks(rotation=55)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks.strftime("%Y-%m-%d"))
    plt.plot(
        df.index,
        df.iloc[:, col],
        label=df.columns[col],
        # marker='.'
    )
    plt.legend()


# find anomaly1 P1, T1, T2, power
anomaly1_indexes = df.loc[df["T1"] < 400].index
surroundings_anomaly1 = get_surrondings_around_indexes(anomaly1_indexes)
draw_figure_axe_by_feature(df, surroundings_anomaly1)

# find anomaly2 Ta3
anomaly2_indexes = (
    df["Ta3"]
    .loc[
        (df.index > pd.Timestamp(2004, 2, 26))
        & (df.index < pd.Timestamp(2004, 2, 28, 23, 50))
    ]
    .index
)
surroundigs_anomaly2 = get_surrondings_around_indexes(anomaly2_indexes)
draw_figure_axe_by_feature(df, anomaly2_indexes)

msno.heatmap(df)

# data corelation
df.corr()
df.corr("spearman")

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True)

# pairplot from pandas - simple & fast
colors = ["blue", "green", "yellow", "red"]
v_1_values = df["v_1"].value_counts().index
feature_v_1_values_to_color = dict(zip(v_1_values, colors))
pd.plotting.scatter_matrix(df, figsize=(20, 15))

# colored by v_1 corelations similar to the upper one
# NOTICE! - it takes 10 minutes to load
sns.pairplot(df, hue="v_1")

# colored by v_1 plot of power
# NOTICE! - it takes 10 minutes to load
fig, ax = plt.subplots(figsize=(8, 6))
for key, group in df.groupby(by="v_1"):
    plt.plot(
        df.index,
        group["power"],
        c=feature_v_1_values_to_color[key],
        label=key,
    )
plt.legend()
