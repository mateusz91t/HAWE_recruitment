import pandas as pd, numpy as np
from matplotlib import pyplot as plt, colors

# plt.style.use("bmh")
# plt.style.available

from sklearn.linear_model import LinearRegression

import random
from functions_and_data_cleaning_dir.additional_functions import (
    prepare_data_frame,
)


df = pd.read_csv("df_task_final.csv")
df = prepare_data_frame(df)

(
    df.index
    == pd.date_range(start=df.index.min(), end=df.index.max(), freq="min")
).all()


##
# one hot encoding - change to continuous values and divide data to X, y
v_1_types = df["v_1"].unique()
v_1_categories = dict(
    {v_1_types[x]: (x + 1.0) * 50 for x in range(len(v_1_types))}
)
v_1_categories

df["v_1"] = df["v_1"].apply(lambda x: v_1_categories[x])  # OHE
df["v_1"] = df["v_1"].astype("int16")
# df["v_1"] = df["v_1"] * 50
df.info()
##
# color_list = list(colors.CSS4_COLORS.keys())
fig, ax = plt.subplots(figsize=(10, 10))
plt.xticks(rotation=45)
for col in range(df.columns.size):
    # color = random.choice(color_list)
    # color_list.remove(color)
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
        df.iloc[:, col],
        label=df.columns[col],
        # color=color
    )
    plt.legend()

##


# one hot encoding - change to continuous values and divide data to X, y
v_1_types = df["v_1"].unique()
v_1_categories = dict({v_1_types[x]: x + 1.0 for x in range(len(v_1_types))})
v_1_categories


X, y = df.iloc[:, :-1], df.iloc[:, -1]  # without a Timestamp, how with it?
X["v_1"] = X["v_1"].apply(lambda x: v_1_categories[x])  # OHE
X
y

# model
lr = LinearRegression()
lr.fit(X, y)
lr.score(X, y)
# lr.
