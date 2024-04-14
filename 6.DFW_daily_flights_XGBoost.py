# %% Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
# from sklearn.pipeline import Pipeline
# from category_encoders.target_encoder import TargetEncoder

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

# %% Load data
DAILY_DATA_PATH = "data.v3/daily" 

df = pd.read_parquet(os.path.join(DAILY_DATA_PATH, "daily_flights_and_weather_merged.parquet"))

# Flights column groups
flights_terminal_cols = ['flights_arr_A', 'flights_arr_B', 'flights_arr_C', 'flights_arr_D', 'flights_arr_E',
                         'flights_dep_A', 'flights_dep_B', 'flights_dep_C', 'flights_dep_D', 'flights_dep_E']

flights_non_terminal_cols = ['flights_total', 'flights_cancel', 'flights_delay', 'flights_ontime',
                             'flights_arr_ontime', 'flights_arr_delay', 'flights_arr_cancel',
                             'flights_dep_ontime', 'flights_dep_delay', 'flights_dep_cancel']

flights_percentage_cols = ['flights_cancel_pct', 'flights_delay_pct', 'flights_ontime_pct',
                            'flights_arr_delay_pct', 'flights_arr_ontime_pct', 'flights_arr_cancel_pct',
                            'flights_dep_delay_pct', 'flights_dep_ontime_pct', 'flights_dep_cancel_pct']

# Date column groups
date_cols = ['date', 'covid', 'ordinal_date', 'year', 'month', 'day_of_month', 'day_of_week', 'season', 'holiday', 'halloween', 'xmas_eve', 'new_years_eve', 'jan_2', 'jan_3', 'day_before_easter', 'days_until_xmas', 'days_until_thanksgiving', 'days_until_july_4th', 'days_until_labor_day', 'days_until_memorial_day']

# Weather column groups
weather_cols = ['wx_temperature_max', 'wx_temperature_min', 'wx_apcp', 'wx_prate', 'wx_asnow', 'wx_frozr', 'wx_vis', 'wx_gust', 'wx_maxref', 'wx_cape', 'wx_lftx', 'wx_wind_speed', 'wx_wind_direction']

# Lag column groups
lag_cols =  ['flights_cancel_lag_1', 'flights_cancel_lag_2', 'flights_cancel_lag_3', 'flights_cancel_lag_4', 'flights_cancel_lag_5', 'flights_cancel_lag_6', 'flights_cancel_lag_7',
             'flights_delay_lag_1', 'flights_delay_lag_2', 'flights_delay_lag_3', 'flights_delay_lag_4', 'flights_delay_lag_5', 'flights_delay_lag_6', 'flights_delay_lag_7',
             'flights_ontime_lag_1', 'flights_ontime_lag_2', 'flights_ontime_lag_3', 'flights_ontime_lag_4', 'flights_ontime_lag_5', 'flights_ontime_lag_6', 'flights_ontime_lag_7',]


# print("Unique data types in df", df.dtypes.value_counts(), sep = '\n')

# # Identify categorical and numeric columns in df
# categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
# numeric_cols = df.select_dtypes(include = ['float64', 'float32', 'int32', 'int64']).columns.tolist()
# num_features = df.shape[1]

# print(f"\nCategorical columns: {categorical_cols}")
# print(f"Numeric columns: {numeric_cols}")
# print(f"\nAll columns accounted for: {len(categorical_cols) + len(numeric_cols) == num_features}")

# %% Split Data
# Select training features
train_features = date_cols + weather_cols + lag_cols

# Create X and y
X = df[train_features].drop('date', axis=1)
y = df[flights_non_terminal_cols + flights_percentage_cols]

print(X.columns.tolist())
print("\nTarget columns\n", y.head())

# Split data into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, shuffle = True, random_state=42)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, shuffle = True, random_state=42)

# Print shapes
print("X_train_full shape:", X_train_full.shape)
print("y_train_full shape:", y_train_full.shape)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_Test shape:", X_test.shape)

# %% XGB With Default Hyperparameters

# Data Preprocessing
X_train = pd.get_dummies(X_train)
X_val = pd.get_dummies(X_val).reindex(columns = X_train.columns, fill_value=0)
X_test = pd.get_dummies(X_test).reindex(columns = X_train.columns, fill_value=0)

dtrain = xgb.DMatrix(X_train, label=y_train['flights_ontime'])
dval = xgb.DMatrix(X_val, label=y_val['flights_ontime'])
dtest = xgb.DMatrix(X_test, label=y_test['flights_ontime'])

# XGB
param = {'max_depth': 5, 
         'eta': 0.01, 
         'objective': 'reg:squarederror',
         'eval_metric': ['rmse', 'mae', 'mape'],
         'seed': 42}

evallist = [(dtrain, 'train'), (dval, 'eval')]

# Train
num_round = 1000
bst = xgb.train(param, dtrain, num_round, evallist, 
                early_stopping_rounds=50, verbose_eval=10)

# Validation
y_val['flights_ontime_pred'] = bst.predict(dval)
mae = mean_absolute_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])
mse = mean_squared_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])
mape = mean_absolute_percentage_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])

print("VALIDATION")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Percentage Error: {mape}")


# Performance
y_train_all = pd.concat([y_train, y_val], axis=0)
plt.figure(figsize=(15, 5))
plt.scatter(y_train_all.index, y_train_all['flights_ontime'], color='blue', alpha = 0.5, label='Actual Flights Ontime')
plt.scatter(y_train_all.index, y_train_all['flights_ontime_pred'], color='red', alpha = 0.5, label='Predicted Flights Ontime')
plt.xlabel('Date')
plt.ylabel('Flights Ontime')
plt.title('Actual and Predicted Flights Ontime')
plt.legend()
plt.show()


# Feature Importance
xgb.plot_importance(bst, max_num_features=15)
plt.show()

# %% Bayesian Hyperparameter tuning (skopt and gp_minimize)

# Define the space of hyperparameters to search
search_spaces = [
    Integer(2, 8, name='max_depth'),
    Real(0.001, 1.0, 'log-uniform', name='learning_rate'),
    Real(0.5, 1.0, name='subsample'),
    Real(0.5, 1.0, name='colsample_bytree'),
    Real(0.5, 1.0, name='colsample_bylevel'),
    Real(0.5, 1.0, name='colsample_bynode'),
    Integer(100, 1000, name='num_boost_round'),
    Real(0, 10, 'uniform', name='min_child_weight'),
    Real(0.0, 10, name='reg_alpha'),
    Real(0.0, 10, name='reg_lambda'),
    Real(0.0, 10, name='gamma')
]

# Define the objective function
@use_named_args(search_spaces)
def objective(**params):
    num_boost_round = params.pop('num_boost_round')
    params['objective'] = 'reg:squarederror'
    params['eval_metric'] = 'mae'
    params['seed'] = 42

    cv_results = xgb.cv(
        params=params,
        dtrain=xgb.DMatrix(X_train, label=y_train['flights_ontime']),
        num_boost_round=num_boost_round,
        nfold=3,
        stratified=False,
        early_stopping_rounds=50,
        seed=42
    )

    # Extract the best score
    best_score = cv_results['test-mae-mean'].min()
    return best_score

# Use gp_minimize to find the best hyperparameters using Bayesian optimization
result = gp_minimize(
    func=objective,
    dimensions=search_spaces,
    n_calls=20,
    random_state=42
)

print("Best Hyperparameters: ", result.x)
print("Best Score: ", result.fun)

# %% Train and Save Best Model

best_params = dict(zip([param.name for param in search_spaces], result.x))
best_params['objective'] = 'reg:squarederror'
best_params['eval_metric'] = 'mae'
best_params['seed'] = 42

best_reg = xgb.train(best_params, 
                     dtrain, num_round, 
                     evallist, 
                     early_stopping_rounds=50, 
                     verbose_eval=10)   

import joblib
joblib.dump(best_reg, 'models/flights_ontime/xgb_model.pkl')

# %% Load & Validate Best Model
best_reg = joblib.load('models/flights_ontime/xgb_model.pkl')

# Validation
y_val['flights_ontime_pred'] = best_reg.predict(dval)
mae = mean_absolute_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])
mse = mean_squared_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])
mape = mean_absolute_percentage_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])

print("VALIDATION METRICS")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Percentage Error: {mape}")

# %% Best Model Performance
y_train_all = pd.concat([y_train, y_val], axis=0)
y_train_all.index = pd.to_datetime(y_train_all.index)

plt.figure(figsize=(15, 5))
plt.scatter(y_train_all.index, y_train_all['flights_ontime'], color='blue', alpha = 0.5, label='Actual Flights Ontime')
plt.scatter(y_train_all.index, y_train_all['flights_ontime_pred'], color='red', alpha = 0.5, label='Predicted Flights Ontime')
plt.xlabel('Date')
plt.ylabel('Flights Ontime')
plt.title('Actual and Predicted Flights Ontime')
plt.legend()
plt.show()

# Actual and Predicted Flights Ontime in December 2020
plt.figure(figsize=(15, 5))
plt.scatter(y_train_all.index, y_train_all['flights_ontime'], color='blue', alpha = 0.5, label='Actual Flights Ontime')
plt.scatter(y_train_all.index, y_train_all['flights_ontime_pred'], color='red', alpha = 0.5, label='Predicted Flights Ontime')
plt.xlabel('Date')
plt.ylabel('Flights Ontime')
plt.title('Actual and Predicted Flights Ontime')
plt.legend()
plt.xlim(pd.to_datetime('2020-12-01'), pd.to_datetime('2020-12-31'))
plt.show()

# Feature Importance
xgb.plot_importance(best_reg, max_num_features=15)
plt.show()

# %%
# NEXT STEPS
# - Plot actual vs predicted for three, week-long intervals
# - Update 0.DFW_daily_flights_EDA.py to include xgb results
# - Train on Kestrel