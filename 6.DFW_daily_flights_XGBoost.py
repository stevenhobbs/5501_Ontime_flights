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
from sklearn.pipeline import Pipeline


from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Integer
from skopt.plots import plot_convergence

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
"""
Proprocessing notes: 
Sklearn's OneHotEncoder returned better validation metrics compared to category_encoder's TargetEncoder. 
TargetEncoder is recommended for XGBoost with high cardinality categorical features, because it does not 
increase data dimensionality. Instead of creating columns for each categorical value, it replaces 
categorical values with a smoothed mean, a weighted average between the overall target mean and the 
category-specific mean. TargetEncoder may have underperformed compared to OneHotEncoder, because
the categorical features in this dataset are low cardinality.

xgb.DMatrix objects offer memory and speed advantagees compared to pandas DataFrames or numpy arrays, 
especially for sparse datasets. DMatrix objects are optimized for distributed computing and allow for 
more detailed configuration of parameters and optimization settings. However, DMatrix objects are not
compatible with sklearn's ColumnTransformer, which is why we convert the transformed data back to a
pandas DataFrame. They are also not compatible with Sklearn's BayesSearchCV, which is why we use skopt
and gp_minimize for hyperparameter tuning. gp_minimize also uses Bayesian optimization to find the best
hyperparameters for the XGBoost model.
"""

X_numeric_cols = X.select_dtypes(include = ['float64', 'float32', 'int32', 'int64']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', X_numeric_cols)
    ])

preprocessor.fit(X_train)

feature_names = preprocessor.get_feature_names_out()

def transform_to_df(preprocessor, X):
    transformed = preprocessor.transform(X)
    df = pd.DataFrame(transformed, columns=feature_names)
    return df

X_train_transformed = transform_to_df(preprocessor, X_train)
X_val_transformed = transform_to_df(preprocessor, X_val)
X_test_transformed = transform_to_df(preprocessor, X_test)
X_full_transformed = transform_to_df(preprocessor, X)

dtrain = xgb.DMatrix(X_train_transformed, label = y_train['flights_ontime'])
dval = xgb.DMatrix(X_val_transformed, label = y_val['flights_ontime'])
dtest = xgb.DMatrix(X_test_transformed, label = y_test['flights_ontime'])
dfull = xgb.DMatrix(X_full_transformed, label = y['flights_ontime'])

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

"""
search_spaces is a list of skopt.space objects that define the search space for the hyperparameters. The search space
specifies the type of hyperparameter (e.g. integer, real) and the range of values to search. The search space is 
defined for the following hyperparameters:
- max_depth: The maximum depth of each tree
- learning_rate: The learning rate of the model
- subsample: The fraction of samples used to fit each tree
- colsample_bytree: The fraction of features used to fit each tree
- colsample_bylevel: The fraction of features used to fit each level of a tree
- colsample_bynode: The fraction of features used to fit each node of a tree
- num_boost_round: The number of boosting rounds (i.e. iterations)
- min_child_weight: The minimum number of instances needed in a child
- reg_alpha: L1 regularization term on weights
- reg_lambda: L2 regularization term on weights
- gamma: Minimum loss reduction required to partition a leaf node
"""
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

"""
Note:
Without the @use_named_args skopt decorator, gp_minimize would pass to the objective function an un-named list of values obtained 
by sampling search_spaces. The decorator adds functionality to the `objective` function, which involves combining the unnamed list 
of values chosen by `gp_minimize` with the names from `search_spaces` into key:value pairs that become the `params` dictionary. The 
params dictionary is then passed to the objective function, allowing the function to access the hyperparameters by name. gp_minimize
uses Bayesian optimization to find the best score ('mae') by searching the hyperparameter space defined in search_spaces. The result
is the best hyperparameters for the XGBoost model.
"""
# Define the objective function
@use_named_args(dimensions = search_spaces)
def objective(**params):
    num_boost_round = params.pop('num_boost_round')
    params['objective'] = 'reg:squarederror'
    params['eval_metric'] = 'mae'
    params['seed'] = 42

    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        nfold=5,
        stratified=False,
        early_stopping_rounds=50,
        seed=42
    )

    # Extract the best score
    best_score = cv_results['test-mae-mean'].min()
    return best_score

# gp_minimize will find hyperparameters that minimize the objective function using Bayesian optimization
result = gp_minimize(
    func=objective,
    dimensions=search_spaces,
    n_calls=200,
    random_state=42
)

plot_convergence(result)
print("Best Hyperparameters: ", result.x)
print("Best Score: ", result.fun)


# %% Train and Save with Best Hyperparameters
# Train with best hyperparameters
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