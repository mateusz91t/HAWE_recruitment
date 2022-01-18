import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from additional_functions import prepare_data_frame

# read data
csvfile = pd.read_csv("df_task_final.csv")
# clean
csvfile.info()
csvfile.describe()
csvfile["v_1"].value_counts()
csvfile = prepare_data_frame(csvfile)

# charts
columns_indexes = [*range(1, 12), 13]
fig, ax = plt.subplots(4, 3, figsize=(20, 15))
for plot_position in range(12):
    column_name = csvfile.columns[columns_indexes[plot_position]]
    plt.subplot(4, 3, plot_position + 1)
    plt.plot(csvfile[column_name])
    plt.title(column_name)

# find anomaly1 P1, T1, T2, power
anomaly1_indexes = csvfile.loc[
    (csvfile["P1"] < -230) & (csvfile.index.isin([*range(4500, 32000)]))
].index

anomaly1_surroundings = [
    *range(anomaly1_indexes[0] - 10, anomaly1_indexes[0]),
    *anomaly1_indexes,
    *range(anomaly1_indexes[-1] + 1, anomaly1_indexes[-1] + 10),
]
csvfile.iloc[anomaly1_surroundings]

# anomaly1 P1, T1, T2, power chart
columns_indexes = [*range(1, 12), 13]
fig, ax = plt.subplots(4, 3, figsize=(20, 15))
for plot_position in range(12):
    column_name = csvfile.columns[columns_indexes[plot_position]]
    plt.subplot(4, 3, plot_position + 1)
    plt.plot(csvfile.iloc[anomaly1_surroundings][column_name])
    plt.title(column_name)


# find anomaly2 Ta3
anomaly1_indexes = csvfile.loc[
    (csvfile["P1"] < -230) & (csvfile.index.isin([*range(4500, 32000)]))
].index

anomaly1_surroundings = [
    *range(anomaly1_indexes[0] - 10, anomaly1_indexes[0]),
    *anomaly1_indexes,
    *range(anomaly1_indexes[-1] + 1, anomaly1_indexes[-1] + 10),
]
csvfile.iloc[anomaly1_surroundings]

# anomaly2 Ta3 chart
columns_indexes = [*range(1, 12), 13]
fig, ax = plt.subplots(4, 3, figsize=(20, 15))
for plot_position in range(12):
    column_name = csvfile.columns[columns_indexes[plot_position]]
    plt.subplot(4, 3, plot_position + 1)
    plt.plot(csvfile.iloc[anomaly1_surroundings][column_name])
    plt.title(column_name)


# data corelation
colors = ["blue", "green", "yellow", "red"]
v_1_values = csvfile["v_1"].value_counts().index
feature_v_1_values_to_color = dict(zip(v_1_values, colors))
pd.plotting.scatter_matrix(csvfile, figsize=(20, 15))

csvfile.corr()
csvfile.corr("spearman")

# colored by v_1 corelations similar to the upper one
# NOTICE! - it takes 10 minutes to load
sns.pairplot(csvfile, hue="v_1")

# colored by v_1 plot of power
# NOTICE! - it takes 10 minutes to load
fig, ax = plt.subplots(figsize=(8, 6))
for key, group in csvfile.groupby(by="v_1"):
    plt.plot(
        group["Timestamp"],
        group["power"],
        c=feature_v_1_values_to_color[key],
        label=key,
    )
plt.legend()
