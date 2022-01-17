import pandas as pd
from matplotlib import pyplot as plt

# read data
csvfile = pd.read_csv('df_task_final.csv')
# clean
csvfile['Ta4'] = csvfile['Ta4'].astype('str')
csvfile['Ta4'] = csvfile['Ta4'].apply(lambda x: x.replace(',', '.'))
csvfile['Ta4'] = csvfile['Ta4'].astype('float32')
csvfile.info()
csvfile.describe()
csvfile['v_1'].value_counts()

# charts
columns_indexes = [*range(1, 12), 13]
fig, ax = plt.subplots(4, 3, figsize=(20, 15))
for plot_position in range(12):
    column_name = csvfile.columns[columns_indexes[plot_position]]
    plt.subplot(4, 3, plot_position + 1)
    plt.plot(csvfile[column_name])
    plt.title(column_name)

# find anomaly
anomaly_indexes = csvfile.loc[
    (csvfile['P1'] < -230)
    & (csvfile.index.isin([*range(4500, 32000)]))
].index

anomaly_surroundings = [
    *range(anomaly_indexes[0]-10, anomaly_indexes[0]),
    *anomaly_indexes,
    *range(anomaly_indexes[-1] + 1, anomaly_indexes[-1]+10)
]
csvfile.iloc[anomaly_surroundings]

# anomaly
columns_indexes = [*range(1, 12), 13]
fig, ax = plt.subplots(4, 3, figsize=(20, 15))
for plot_position in range(12):
    column_name = csvfile.columns[columns_indexes[plot_position]]
    plt.subplot(4, 3, plot_position + 1)
    plt.plot(csvfile.iloc[anomaly_surroundings][column_name])
    plt.title(column_name)


# data corelation
pd.plotting.scatter_matrix(csvfile, figsize=(20, 15))
