import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

from math import ceil

from functions_and_data_cleaning_dir.additional_functions import (
    prepare_data_frame,
    draw_figure_axe_by_feature,
)


# read data
df = pd.read_csv("df_task_final.csv")
# clean
df.info()
df.describe()
df["v_1"].value_counts()
df = prepare_data_frame(df)
# one hot encoding - change to continuous values and divide data to X, y
v_1_types = df["v_1"].unique()
v_1_categories = dict(
    {v_1_types[x]: (x + 1.0) * 50 for x in range(len(v_1_types))}
)
df["v_1"] = df["v_1"].apply(lambda x: v_1_categories[x])  # OHE
df["v_1"] = df["v_1"].astype("int16")

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
anomaly1_indexes = df.loc[df["Ta3"] < 0].index
surroundings_anomaly1 = get_surrondings_around_indexes(anomaly1_indexes)
# anomaly1 P1, T1, T2, power chart
draw_figure_axe_by_feature(df, surroundings_anomaly1)

# find anomaly2 Ta3
anomaly2_indexes = df["Ta3"].loc[
    (df.index > pd.Timestamp(2004, 2, 26))
    & (df.index < pd.Timestamp(2004, 2, 28, 23, 50))
].index
surroundigs_anomaly2 = get_surrondings_around_indexes(anomaly2_indexes)
# anomaly2 Ta3 chart
draw_figure_axe_by_feature(df, anomaly2_indexes)


# data corelation
colors = ["blue", "green", "yellow", "red"]
v_1_values = df["v_1"].value_counts().index
feature_v_1_values_to_color = dict(zip(v_1_values, colors))
pd.plotting.scatter_matrix(df, figsize=(20, 15))

df.corr()
df.corr("spearman")

fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot=True)

# colored by v_1 corelations similar to the upper one
# NOTICE! - it takes 10 minutes to load
sns.pairplot(df, hue="v_1")

# colored by v_1 plot of power
# NOTICE! - it takes 10 minutes to load
fig, ax = plt.subplots(figsize=(8, 6))
for key, group in df.groupby(by="v_1"):
    plt.plot(
        group["Timestamp"],
        group["power"],
        c=feature_v_1_values_to_color[key],
        label=key,
    )
plt.legend()
