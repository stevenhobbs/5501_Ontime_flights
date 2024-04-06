# %% Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder

from xgboost import XGBRegressor, plot_importance, plot_tree
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

# %% XGB Training Pipeline

X_train = pd.get_dummies(X_train)
X_val = pd.get_dummies(X_val).reindex(columns = X_train.columns, fill_value=0)
X_test = pd.get_dummies(X_test).reindex(columns = X_train.columns, fill_value=0)

dtrain = xgb.DMatrix(X_train, label=y_train['flights_ontime'])
dval = xgb.DMatrix(X_val, label=y_val['flights_ontime'])
dtest = xgb.DMatrix(X_test, label=y_test['flights_ontime'])

param = {'max_depth': 5, 
         'eta': 0.01, 
         'objective': 'reg:squarederror',
         'eval_metric': ['rmse', 'mae', 'mape'],
         'seed': 42}

evallist = [(dtrain, 'train'), (dval, 'eval')]

num_round = 1000
bst = xgb.train(param, dtrain, num_round, evallist, 
                early_stopping_rounds=50, verbose_eval=10)

xgb.plot_importance(bst, max_num_features=15)

# %% Sklearn-like Training API
reg = XGBRegressor(n_estimators = 1000, 
                   learning_rate = 0.01, 
                   max_depth = 5, 
                   random_state = 42, 
                   objective = 'reg:squarederror')

reg.fit(X_train, y_train['flights_ontime'],
        eval_set = [(X_train, y_train['flights_ontime']), (X_val, y_val['flights_ontime'])],
        early_stopping_rounds = 50,
        verbose = True)

plot_importance(reg, max_num_features=15)
# %% Prediction
y_val['flights_ontime_pred'] = reg.predict(X_val)
y_train_all = pd.concat([y_train, y_val], axis=0)

plt.figure(figsize=(15, 5))
plt.scatter(y_train_all.index, y_train_all['flights_ontime'], color='blue', alpha = 0.5, label='Actual Flights Ontime')
plt.scatter(y_train_all.index, y_train_all['flights_ontime_pred'], color='red', alpha = 0.5, label='Predicted Flights Ontime')
plt.xlabel('Date')
plt.ylabel('Flights Ontime')
plt.title('Actual and Predicted Flights Ontime')
plt.legend()
plt.show()

# %% Evaluation
mae = mean_absolute_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])
mse = mean_squared_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])
mape = mean_absolute_percentage_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Percentage Error: {mape}")

# %% Hyperparameter tuning (skopt)
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

search_spaces = {
    'reg__max_depth': Integer(2, 8),

    'reg__learning_rate': Real(0.001, 1.0, 'log-uniform'),
    'reg__subsample': Real(0.5, 1.0),
    'reg__colsample_bytree': Real(0.5, 1.0),
    'reg__colsample_bylevel': Real(0.5, 1.0),
    'reg__colsample_bynode': Real(0.5, 1.0),

    # 'reg__n_estimators': Integer(100, 1000),
    # 'reg__min_child_weight': Real(0, 10, 'uniform'),
    # 'reg__max_delta_step': Real(0, 10, 'uniform'),

    'reg__reg_alpha': Real(0.0, 10,),
    'reg__reg_lambda': Real(0.0, 10),
    'reg__gamma': Real(0.0, 10)
}


