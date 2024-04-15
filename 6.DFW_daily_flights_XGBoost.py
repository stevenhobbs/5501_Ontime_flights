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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
# from category_encoders.target_encoder import TargetEncoder

from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

# %% Load data
DAILY_DATA_PATH = "data.v3/daily" 

df = pd.read_parquet(os.path.join(DAILY_DATA_PATH, "daily_flights_and_weather_merged.parquet"))
df = df.drop('date', axis=1)

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
date_cols = ['covid', 'ordinal_date', 'year', 'month', 'day_of_month', 'day_of_week', 'season', 'holiday', 'halloween', 'xmas_eve', 'new_years_eve', 'jan_2', 'jan_3', 'day_before_easter', 'days_until_xmas', 'days_until_thanksgiving', 'days_until_july_4th', 'days_until_labor_day', 'days_until_memorial_day']

# Weather column groups
weather_cols = ['wx_temperature_max', 'wx_temperature_min', 'wx_apcp', 'wx_prate', 'wx_asnow', 'wx_frozr', 'wx_vis', 'wx_gust', 'wx_maxref', 'wx_cape', 'wx_lftx', 'wx_wind_speed', 'wx_wind_direction']

# Lag column groups
lag_cols =  ['flights_cancel_lag_1', 'flights_cancel_lag_2', 'flights_cancel_lag_3', 'flights_cancel_lag_4', 'flights_cancel_lag_5', 'flights_cancel_lag_6', 'flights_cancel_lag_7',
             'flights_delay_lag_1', 'flights_delay_lag_2', 'flights_delay_lag_3', 'flights_delay_lag_4', 'flights_delay_lag_5', 'flights_delay_lag_6', 'flights_delay_lag_7',
             'flights_ontime_lag_1', 'flights_ontime_lag_2', 'flights_ontime_lag_3', 'flights_ontime_lag_4', 'flights_ontime_lag_5', 'flights_ontime_lag_6', 'flights_ontime_lag_7',]


print("Unique data types in df", df.dtypes.value_counts(), sep = '\n')

# # Identify categorical and numeric columns in df
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include = ['float64', 'float32', 'int32', 'int64']).columns.tolist()
num_features = df.shape[1]

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numeric columns: {numeric_cols}")
print(f"\nAll columns accounted for: {len(categorical_cols) + len(numeric_cols) == num_features}")

# %% Split Data
# Select training features
train_features = date_cols + weather_cols + lag_cols

# Create X and y
X = df[train_features]
y = df[flights_non_terminal_cols + flights_percentage_cols]

print(X.columns.tolist())
print("\nTarget columns\n", y.head())

# Split data into train and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, shuffle = True, random_state=42)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, shuffle = True, random_state=42)

# Print shapes
print("\nX datasets:")
print("X shape:", X.shape)
print("X_train_full shape:", X_train_full.shape)
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_Test shape:", X_test.shape)
print("\ny datasets:")
print("y shape:", y.shape)
print("y_train_full shape:", y_train_full.shape)
print("y_train shape:", y_train.shape)
print("y_val shape:", y_val.shape)
print("y_Test shape:", y_test.shape)

# %% Data Preprocessing
from sklearn.base import BaseEstimator, TransformerMixin


X_numeric_cols = X.select_dtypes(include = ['float64', 'float32', 'int32', 'int64']).columns.tolist()

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer
        self.get_feature_names_out = None

    def fit(self, X, y=None):
        self.transformer.fit(X[0], y)
        self.get_feature_names_out = self.transformer.get_feature_names_out
        return self

    def transform(self, X):
        return (self.transformer.transform(X[0]), X[1])

class ArraytoDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X[0], columns = self.feature_names)
        return (df, X[1])
    
class DataFrameToDMatrix(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return xgb.DMatrix(X[0], label = X[1])
    
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', X_numeric_cols)
    ])

pipeline = Pipeline([
    ('custom_preprocessor', CustomPreprocessor(transformer=preprocessor)),
    ('array_to_df', ArraytoDataFrame(feature_names = None)),
    ('df_to_dmatrix', DataFrameToDMatrix())
])

pipeline.fit((X_train, y_train['flights_ontime']))

# Set the feature names for the array_to_df transformer
feature_names = pipeline.named_steps['custom_preprocessor'].get_feature_names_out()
pipeline.named_steps['array_to_df'].feature_names = feature_names

# Transform datasets
dtrain = pipeline.transform((X_train, y_train['flights_ontime']))
dfull = pipeline.transform((X, y['flights_ontime']))
dval = pipeline.transform((X_val, y_val['flights_ontime']))
dtest = pipeline.transform((X_test, y_test['flights_ontime']))


# X_train = preprocessor.fit_transform(X_train)
# X_train = pd.DataFrame(X_train, columns = preprocessor.get_feature_names_out())

# X_full = preprocessor.transform(X)
# X_full = pd.DataFrame(X_full, columns = preprocessor.get_feature_names_out())

# X_val = preprocessor(X_val)
# X_val = pd.DataFrame(X_val, columns = preprocessor.get_feature_names_out())

# X_test = preprocessor(X_test)
# X_test = pd.DataFrame(X_test, columns = preprocessor.get_feature_names_out())

# # Create DMatrix for faster XGB performance
# X_DMatrix = xgb.DMatrix(X_full, label=y['flights_ontime'])
# dtrain = xgb.DMatrix(X_train, label=y_train['flights_ontime'])
# dval = xgb.DMatrix(X_val, label=y_val['flights_ontime'])
# dtest = xgb.DMatrix(X_test, label=y_test['flights_ontime'])

# %% XGB With Default Hyperparameters
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
        dtrain=dtrain,
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

joblib.dump(best_reg, 'models/flights_ontime/xgb_model.pkl')

# %% Load & Validate Best Model
best_reg = joblib.load('models/flights_ontime/xgb_model.pkl')
results_dir = 'model_output/flights_ontime'
os.makedirs(results_dir, exist_ok=True)

# Validation
y_val['flights_ontime_pred'] = best_reg.predict(dval)
val_performance = {}
val_performance['XGBoost'] = {
    'MAE': mean_absolute_error(y_val['flights_ontime'], y_val['flights_ontime_pred']),
    'MSE': mean_squared_error(y_val['flights_ontime'], y_val['flights_ontime_pred']),
    'MAPE': mean_absolute_percentage_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])
}
val_performance_df = pd.DataFrame(val_performance).T
val_performance_df.to_csv(results_dir + '/xgb_val_performance.csv')

print("Validation Performance")
print(val_performance_df.round(2))


# mae = mean_absolute_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])
# mse = mean_squared_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])
# mape = mean_absolute_percentage_error(y_val['flights_ontime'], y_val['flights_ontime_pred'])

# print("VALIDATION METRICS")
# print(f"Mean Absolute Error: {mae}")
# print(f"Mean Squared Error: {mse}")
# print(f"Mean Absolute Percentage Error: {mape}")

# %% Best Model Performance
y['flights_ontime_pred'] = best_reg.predict(dfull)

plt.figure(figsize=(15, 5))
plt.scatter(y.index, y['flights_ontime'], color='blue', alpha = 0.5, label='Actual Flights Ontime')
plt.scatter(y.index, y['flights_ontime_pred'], color='red', alpha = 0.25, label='Predicted Flights Ontime')
plt.xlabel('Date')
plt.ylabel('Flights Ontime')
plt.title('Actual and Predicted Flights Ontime')
plt.legend()
plt.show()

# Actual and Predicted Flights Ontime by Month over the most recent 12 months
for i in range(13):
    start = y.index[-1] - pd.DateOffset(months=i)
    end = y.index[-1] - pd.DateOffset(months=i-1) - pd.DateOffset(days=1)
    plt.figure(figsize=(15, 5))
    plt.scatter(y[start:end].index, y[start:end]['flights_ontime'], color='blue', alpha = 0.5, label='Actual Flights Ontime')
    plt.scatter(y[start:end].index, y[start:end]['flights_ontime_pred'], color='red', alpha = 0.5, label='Predicted Flights Ontime')
    plt.ylabel('Flights Ontime')
    plt.title(f'Actual and Predicted Flights Ontime - {start.strftime("%B")}, {start.year}')
    plt.legend()
    plt.show()

# plt.figure(figsize=(15, 5))
# plt.scatter(y.index, y['flights_ontime'], color='blue', alpha = 0.5, label='Actual Flights Ontime')
# plt.scatter(y.index, y['flights_ontime_pred'], color='red', alpha = 0.5, label='Predicted Flights Ontime')
# plt.xlabel('Date')
# plt.ylabel('Flights Ontime')
# plt.title('Actual and Predicted Flights Ontime')
# plt.legend()
# plt.xlim(pd.to_datetime('2020-12-01'), pd.to_datetime('2020-12-31'))
# plt.show()

# Feature Importance
xgb.plot_importance(best_reg, max_num_features=15)
plt.show()

# %%
# NEXT STEPS
# - Update 0.DFW_daily_flights_EDA.py to include xgb results
# - Train on Kestrel