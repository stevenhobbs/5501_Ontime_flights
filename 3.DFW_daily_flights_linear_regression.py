"""
This script trains linear regression models to predict same-day 
flight traffic and forecasted flight traffic 24 hours ahead. Both
models use only current weather and date information (no flight 
traffic lag columns). Quadratic weather features are included and 
elastic net regularization is used to prevent overfitting.
"""
# %% Libraries
import os
import pandas as pd
pd.set_option('display.width', 600)
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error   
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.exceptions import ConvergenceWarning

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
# %% Import data
DAILY_DATA_PATH = "data.v3/daily" 

# df = pd.read_parquet(os.path.join(DAILY_DATA_PATH, "daily_flights_and_weather_merged.parquet"))
df = pd.read_parquet(DAILY_DATA_PATH + "/daily_flights_and_weather_merged.parquet")


# Flights column groups
flights_terminal_cols = ['flights_arr_A', 'flights_arr_B', 'flights_arr_C', 'flights_arr_D', 'flights_arr_E',
                         'flights_dep_A', 'flights_dep_B', 'flights_dep_C', 'flights_dep_D', 'flights_dep_E']

flights_non_terminal_cols = ['flights_total', 'flights_cancel', 'flights_delay', 'flights_ontime',
                             'flights_arr_ontime', 'flights_arr_delay', 'flights_arr_cancel',
                             'flights_dep_ontime', 'flights_dep_delay', 'flights_dep_cancel']

flights_percentage_cols = ['flights_cancel_pct', 'flights_delay_pct', 'flights_ontime_pct',
                            'flights_arr_delay_pct', 'flights_arr_ontime_pct', 'flights_arr_cancel_pct',
                            'flights_dep_delay_pct', 'flights_dep_ontime_pct', 'flights_dep_cancel_pct']

flights_prediction_cols = flights_non_terminal_cols + flights_percentage_cols
flights_forecast_cols = [f"{col}_next_day" for col in flights_prediction_cols]

# Date column groups
date_cols = ['date', 'covid', 'ordinal_date', 'year', 'month', 'day_of_month', 'day_of_week', 'season', 'holiday', 'halloween', 'xmas_eve', 'new_years_eve', 'jan_2', 'jan_3', 'day_before_easter', 'days_until_xmas', 'days_until_thanksgiving', 'days_until_july_4th', 'days_until_labor_day', 'days_until_memorial_day']

# Weather column groups
weather_cols = ['wx_temperature_max', 'wx_temperature_min', 'wx_apcp', 'wx_prate', 'wx_asnow', 'wx_frozr', 'wx_vis', 'wx_gust', 'wx_maxref', 'wx_cape', 'wx_lftx', 'wx_wind_speed', 'wx_wind_direction']
weather_cols_s2 = ['wx_temperature_max_s2', 'wx_temperature_min_s2', 'wx_apcp_s2', 'wx_prate_s2', 'wx_asnow_s2', 'wx_frozr_s2', 'wx_vis_s2', 'wx_gust_s2', 'wx_maxref_s2', 'wx_cape_s2', 'wx_lftx_s2', 'wx_wind_speed_s2', 'wx_wind_direction_s2']

# Lag column groups
lag_cols =  ['flights_cancel_lag_1', 'flights_cancel_lag_2', 'flights_cancel_lag_3', 'flights_cancel_lag_4', 'flights_cancel_lag_5', 'flights_cancel_lag_6', 'flights_cancel_lag_7',
             'flights_delay_lag_1', 'flights_delay_lag_2', 'flights_delay_lag_3', 'flights_delay_lag_4', 'flights_delay_lag_5', 'flights_delay_lag_6', 'flights_delay_lag_7',
             'flights_ontime_lag_1', 'flights_ontime_lag_2', 'flights_ontime_lag_3', 'flights_ontime_lag_4', 'flights_ontime_lag_5', 'flights_ontime_lag_6', 'flights_ontime_lag_7']
# %% Random data split 80:10:10
# Select training features
train_features = date_cols + weather_cols + weather_cols_s2

# Create X and y
X = df.iloc[:-1][train_features].drop('date', axis=1)
y = df.iloc[:-1][flights_prediction_cols + flights_forecast_cols]

print("\nTarget columns head\n", y.head())
print("\n\nTarget columns tail\n", y.tail())

# Split data into trai_full and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Split train_full into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

# Print shapes
print("\n\nX_train_full shape:", X_train_full.shape)
print("y_train_full shape:", y_train_full.shape)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_Test shape:", X_test.shape)

# %% Preprocessing
categorical_tranformer = make_pipeline(OneHotEncoder(handle_unknown='ignore')) # Some observed holidays may not be in the training data
numeric_transformer = make_pipeline(StandardScaler())

# print value counts of unique data types in X
print(X.dtypes.value_counts())

# Identify categorical and numeric columns in X_train_full
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include = ['float64', 'float32', 'int32', 'int64']).columns.tolist()

# Check that all columns are accounted for
print(f"categorical columns: {categorical_cols}")
print(f"numeric columns: {numeric_cols}")
assert len(categorical_cols) + len(numeric_cols) == X_train_full.shape[1] 
print("All columns are accounted for!")

# Linear regression transformer
LR__transformer = ColumnTransformer(
    transformers=[
        ('cat', categorical_tranformer, categorical_cols),
        ('num', numeric_transformer, numeric_cols)
    ])

# %% LR pipeline with elastic net regularization pipeline

# l1_ration and alpha have default values of 0.5 and 1.0, but will be tuned
# using value ranges set in search_spaces.

LR_pipeline = make_pipeline(
    LR__transformer,
    ElasticNet(alpha=10, 
               l1_ratio=0.5,
               max_iter=10000))


# %% BayesSearchCV tuner
search_spaces = {
    'elasticnet__alpha': Real(1e-6, 1e+1, prior='log-uniform'),
    'elasticnet__l1_ratio': Real(0, 1, prior='uniform')
}

bayes_search = BayesSearchCV(
    LR_pipeline,
    search_spaces,
    n_iter=50,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0,
    return_train_score=True
)
# %% Tune and fit prediction models
LR_prediction_models = {}
convergence_issues = {}
models_dir = "models/daily/LR_prediction"
os.makedirs(models_dir, exist_ok=True)

for target in flights_prediction_cols:
    bayes_search.fit(X_train, y_train[target])

    # Save best model parameters, best alpha, and best model
    best_model = bayes_search.best_estimator_
    best_alpha = best_model.named_steps['elasticnet'].get_params()['alpha']
    best_l1_ratio = best_model.named_steps['elasticnet'].get_params()['l1_ratio']
    LR_prediction_models[f"LR_prediction_{target}"] = bayes_search.best_estimator_

    # Identify convergence issues for the best alpha values and l1_ratio
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        best_model.fit(X_train, y_train[target])
        if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
            convergence_issues[target] = (best_alpha, best_l1_ratio)

# Print convergence issues
if convergence_issues:
    print("Convergence issues:")
    for target, alpha_l1_ratio in convergence_issues.items():
        print(f"{target} did not converge with alpha = {alpha_l1_ratio[0]} and l1_ratio = {alpha_l1_ratio[1]}")
else:
    print("No convergence issues for the best alpha and l1_ratio values of any target")

    # print(f"Best parameters for elastic_net_{target}:\n{grid_search.best_params_}")

# Save best elastic net models
for target, model in LR_prediction_models.items():
    model_path = os.path.join(models_dir, f"{target}.joblib")
    joblib.dump(model, model_path)
    print(f"Saved {target} model to {model_path}")


# %% Tune and fit forecast models
LR_24h_forecast_models = {}
convergence_issues = {}
models_dir = "models/daily/LR_24h_forecasts"
os.makedirs(models_dir, exist_ok=True)

for target in flights_forecast_cols:
    bayes_search.fit(X_train, y_train[target])

    # Save best model parameters, best alpha, and best model
    best_model = bayes_search.best_estimator_
    best_alpha = best_model.named_steps['elasticnet'].get_params()['alpha']
    best_l1_ratio = best_model.named_steps['elasticnet'].get_params()['l1_ratio']
    LR_24h_forecast_models[f"LR_24h_forecast_{target}"] = bayes_search.best_estimator_

    # Identify convergence issues for the best alpha values and l1_ratio
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        best_model.fit(X_train, y_train[target])
        if any(issubclass(warn.category, ConvergenceWarning) for warn in w):
            convergence_issues[target] = (best_alpha, best_l1_ratio)

# Print convergence issues
if convergence_issues:
    print("Convergence issues:")
    for target, alpha_l1_ratio in convergence_issues.items():
        print(f"{target} did not converge with alpha = {alpha_l1_ratio[0]} and l1_ratio = {alpha_l1_ratio[1]}")
else:
    print("No convergence issues for the best alpha and l1_ratio values of any target")

    # print(f"Best parameters for elastic_net_{target}:\n{grid_search.best_params_}")

# Save best elastic net models
for target, model in LR_24h_forecast_models.items():
    model_path = os.path.join(models_dir, f"{target}.joblib")
    joblib.dump(model, model_path)
    print(f"Saved {target} model to {model_path}")
# %% Linear regression prediction results

# Prediction results dictionary
LR_results = pd.DataFrame(columns=['TARGET', 'ALPHA', 'L1L2', 'R2', 'MAE', 'MSE', 'MAPE'])

# Prediction results
for target, model in LR_prediction_models.items():
    target = target.replace("LR_prediction_", "")
    alpha = model.named_steps['elasticnet'].get_params()['alpha']
    l1_ratio = model.named_steps['elasticnet'].get_params()['l1_ratio']
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val[target], y_pred)
    mae = mean_absolute_error(y_val[target], y_pred)
    mse = mean_squared_error(y_val[target], y_pred)
    mape = mean_absolute_percentage_error(y_val[target], y_pred)
    LR_results = LR_results.append({'TARGET': target, 'ALPHA': alpha, 'L1L2': l1_ratio, 
                                    'R2': r2, 'MAE': mae, 'MSE': mse, 'MAPE': mape}, 
                                    ignore_index=True)
    
# Save prediction results to csv
output_dir = "model_output/daily/LR"
os.makedirs(output_dir, exist_ok=True)
LR_results.to_csv(output_dir + "/LR_prediction.csv", index=False)
print(LR_results)

# %% Linear regression forecast results
# Forecast results dictionary
LR_forecast_results = pd.DataFrame(columns=['TARGET', 'ALPHA', 'L1L2', 'R2', 'MAE', 'MSE', 'MAPE'])

# Forecast results
for target, model in LR_24h_forecast_models.items():
    target = target.replace("LR_24h_forecast_", "")
    alpha = model.named_steps['elasticnet'].get_params()['alpha']
    l1_ratio = model.named_steps['elasticnet'].get_params()['l1_ratio']
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val[target], y_pred)
    mae = mean_absolute_error(y_val[target], y_pred)
    mse = mean_squared_error(y_val[target], y_pred)
    mape = mean_absolute_percentage_error(y_val[target], y_pred)
    LR_forecast_results = LR_forecast_results.append({'TARGET': target, 'ALPHA': alpha, 'L1L2': l1_ratio, 
                                    'R2': r2, 'MAE': mae, 'MSE': mse, 'MAPE': mape}, 
                                    ignore_index=True)
    
# Save forecast results to csv
LR_forecast_results.to_csv(output_dir + "/LR_forecast.csv", index=False)
print(LR_forecast_results)
