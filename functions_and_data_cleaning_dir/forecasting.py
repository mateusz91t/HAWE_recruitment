import pandas as pd

from sklearn.linear_model import LinearRegression

from functions_and_data_cleaning_dir.additional_functions import (
    prepare_data_frame,
    one_hot_encoding,
)

# read & clean data
df = pd.read_csv("df_task_final.csv")
df = prepare_data_frame(df)

# Is there each value in a timeseries?
(
    df.index
    == pd.date_range(start=df.index.min(), end=df.index.max(), freq="min")
).all()

# one hot encoding - change to continuous values and divide data to X, y
df = one_hot_encoding(df, "v_1", 50)

# fill NaN
df.loc[df.isnull().sum(axis=1) > 0]
df.fillna(method="ffill", inplace=True)

# divide data to X, y
X, y = df.iloc[:, :-1], df.iloc[:, -1]
X
y

# model
lr = LinearRegression()
lr.fit(X, y)
lr.score(X, y)

sample_to_predict = {
    "P1": -178,
    "T1": 1520,
    "T2": 1520,
    "Ta1": 925,
    "Ta3": -274,
    "Ta4": 930,
    "Ta5": 900,
    "Ta6": 950,
    "Ta7": 1000,
    "Ta8": 1000,
    "Ta_q": 380,
    "v_1": 100.0,
}

sample_to_predict = pd.DataFrame(
    sample_to_predict, index=pd.date_range("2004-03-02 00:00:00", periods=1)
)

predicted_last = lr.predict(X.iloc[-2:-1])
predicted_sample = lr.predict(sample_to_predict)
print(f"predicted_last={predicted_last}\ttarget={y[-1]}")
print(f"predicted_sample={predicted_last}")


# Skforecast: time series forecasting with Python and Scikit-learn
# by Joaquín Amat Rodrigo and Javier Escobar Ortiz,
# available under a Attribution 4.0 International (CC BY 4.0) at
# https://www.cienciadedatos.net/py27-forecasting-series-temporales-python-scikitlearn.html 
# ==============================================================================
from skforecast import ForecasterAutoreg, ForecasterAutoregMultiOutput
from skforecast.model_selection import (
    grid_search_forecaster,
    backtesting_forecaster,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# Split data into train-test
# ==============================================================================
steps = 10500
data_train = df[:-steps]
data_test = df[-steps:]

fig, ax = plt.subplots(figsize=(9, 8))
data_train["power"].plot(ax=ax, label="train")
data_test["power"].plot(ax=ax, label="test")
ax.legend()

# Create and train forecaster
# ==============================================================================
forecaster = ForecasterAutoreg.ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123), lags=6
)

forecaster.fit(y=data_train["power"])

forecaster

# Predictions
# ==============================================================================
predictions = forecaster.predict(steps=steps)

# Comparison - not good result...
fig, ax = plt.subplots(figsize=(15, 15))
data_train["power"].plot(ax=ax, label="train")
data_test["power"].plot(ax=ax, label="test")
predictions.index = data_test["power"].index
predictions.plot(ax=ax, label="predictions")
ax.legend()

# Error
# ==============================================================================
error_mse = mean_squared_error(y_true=data_test["power"], y_pred=predictions)
print(f"Test error (mse): {error_mse}")

# Hyperparameter Grid search
# ==============================================================================
forecaster = ForecasterAutoreg.ForecasterAutoreg(
    regressor=RandomForestRegressor(random_state=123),
    lags=12,  # This value will be replaced in the grid search
)

# Regressor's hyperparameters
param_grid = {"n_estimators": [100, 500], "max_depth": [3, 5, 10]}

# Lags used as predictors
lags_grid = [10, 20]

results_grid = grid_search_forecaster(
    forecaster=forecaster,
    y=data_train["power"],
    param_grid=param_grid,
    lags_grid=lags_grid,
    steps=10,
    refit=True,
    metric="mean_squared_error",
    initial_train_size=int(len(data_train) * 0.5),
    return_best=True,
    verbose=True,
)

# Create and train forecaster with the best hyperparameters
# ==============================================================================
regressor = RandomForestRegressor(max_depth=3, n_estimators=500, random_state=123)

forecaster = ForecasterAutoreg(
                regressor = regressor,
                lags      = 20
             )

forecaster.fit(y=data_train['power'])

# Predictions
# ==============================================================================
predictions = forecaster.predict(steps=steps)

# Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
data_train['power'].plot(ax=ax, label='train')
data_test['power'].plot(ax=ax, label='test')
predictions.plot(ax=ax, label='predictions')
ax.legend();
