#%% Libraries
import os
import datetime
import IPython

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.compose import ColumnTransformer


import tensorflow as tf
from tensorflow.data import Dataset, AUTOTUNE

from keras import layers, models, Sequential, regularizers
from keras.layers import SimpleRNN, Dense, Dropout, Embedding, LSTM, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import timeseries_dataset_from_array
from keras.utils import plot_model
from keras.regularizers import L1, L2, L1L2
from keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

import keras_tuner as kt

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# %% Import data and define column groups
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

# Drop lag columns and date from data
df = df.drop(columns=lag_cols + ['date'])

print("Unique data types in df", df.dtypes.value_counts(), sep = '\n')

# Identify categorical and numeric columns in df
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = df.select_dtypes(include = ['float64', 'float32', 'int32', 'int64']).columns.tolist()
num_features = df.shape[1]

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numeric columns: {numeric_cols}")
print(f"\nAll columns accounted for: {len(categorical_cols) + len(numeric_cols) == num_features}")

# %% Split data sequentially 70-20-10
n = len(df)
train_raw = df[0:int(n*0.7)]
val_raw = df[int(n*0.7):int(n*0.9)]
test_raw = df[int(n*0.9):]

# print data shapes
print(f"Train data preprocessed shape: {train_raw.shape}")
print(f"Validation preprocessed data shape: {val_raw.shape}")
print(f"Test data preprocessed shape: {test_raw.shape}")

# %% Preprocess data for time series
scale_cols = [col for col in numeric_cols if col != 'flights_ontime']

# Fit transformers to the training data
scaler = StandardScaler()
scaler.fit(train_raw[scale_cols])

# Create a scaler to enable inversing scaling of the flights_ontime column
# flights_ontime_scaler = StandardScaler()
# flights_ontime_scaler.fit(train_raw[['flights_ontime']])

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # Some observed holidays may not be in the training data
ohe.fit(train_raw[categorical_cols])
ohe_column_names = ohe.get_feature_names_out(input_features=categorical_cols)

def preprocess(data):
    scaled_features = scaler.transform(data[scale_cols])
    encoded_features = ohe.transform(data[categorical_cols])
    processed_data = pd.DataFrame(np.concatenate([scaled_features, encoded_features, data[['flights_ontime']]], axis=1),
                                  columns = scale_cols + list(ohe_column_names) + ['flights_ontime'])
    return processed_data

train_df = preprocess(train_raw)
val_df = preprocess(val_raw)
test_df = preprocess(test_raw)

print(f"\nNumber of columns before preprocessing: {num_features}")
print(f"Number of features after preprocessing: {train_df.shape[1]}")
# %% WindowGenerator Class
"""
The `WindowGenerator` class stores the train, validate, and test sets (Pandas DataFrames), 
and it handles indexes and offsets for windowing. Several methods are added to this class 
after its creation. When the `split_window` method is added, the class will split data 
windows into separate tensors for features and labels. The `plot` method, produces a plot 
of an example batch showing inputs, labels, and predictions. The `make_dataset` method 
creates TensorFlow timeseries datastes that are batched, windowed, and ready for use in 
modeling.
"""

column_indices = {name: i for i, name in enumerate(train_df.columns)}

class WindowGenerator():
  """
  A class to hold train, validate, and test sets and manage windowing and conversion to TensorFlow time series datasets

  Attributes:
    input_width (int): The number of time steps to include in the input window
    label_width (int): The number of time steps to include in the label window
    shift (int): The number of time steps to shift the label window to create the next input window
    train_df (pd.DataFrame): The training data
    val_df (pd.DataFrame): The validation data
    test_df (pd.DataFrame): The test data
    label_columns (list): The columns to predict

  Methods:
    __init__(input_width, label_width, shift, train_df, val_df, test_df, label_columns)
    __repr__() # Return a string representation of the window
  """
  
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
               
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
  def split_window(self, features):
    """
    Given a window of features, the method splits the features into inputs and labels

    Args:
    features: A window of features

    Returns:
    inputs: A window of input features
    labels: A window of label features
    """

    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels
  
  def plot(self, model=None, plot_col='flights_ontime', max_subplots=3):
    """
    Given a model and a column to plot, the method plots the inputs, labels, and predictions

    Args:
    model: A trained model
    plot_col: The column to plot
    max_subplots: The maximum number of subplots to display

    Returns:
    """

    inputs, labels = self.example

    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))

    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')

        # Plot inputs
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                label='Inputs', marker='.', zorder=-10)

        # Plot labels
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        
        # Plot predictions
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [days]')
    
  def make_dataset(self,data, seed=42):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        seed = seed,
        batch_size=32,)
    
    ds = ds.map(self.split_window)
    return ds
  
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
     return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
    return result
  
#%% Instantiate Window Generators
w28 = WindowGenerator(input_width=28, label_width=28, shift=1,
                      label_columns=['flights_ontime'])

w14 = WindowGenerator(input_width=14, label_width=14, shift=1,
                      label_columns=['flights_ontime'])

w7 = WindowGenerator(input_width=7, label_width=7, shift=1,
                     label_columns=['flights_ontime'])

w3 = WindowGenerator(input_width=3, label_width=3, shift=1,
                     label_columns=['flights_ontime'])

w2 = WindowGenerator(input_width=2, label_width=2, shift=1,
                     label_columns=['flights_ontime'])

w1 = WindowGenerator(input_width=1, label_width=1, shift=1,
                     label_columns=['flights_ontime'])
# %% Demonstrate split window method
example_batch = tf.stack([np.array(train_df[:w7.total_window_size]),
                           np.array(train_df[100:100+w7.total_window_size]),
                           np.array(train_df[200:200+w7.total_window_size])])

example_inputs, example_labels = w7.split_window(example_batch)

print('Window shapes are: (batch size, time steps, features)')
print(f'w7 shape: {example_batch.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')
# %% Example plot inputs & labels
w7.plot()
# %% Baseline Class
"""
The `Baseline` class inherits from `keras.Model` and uses the current value of a 
label to predict a label one step (one day) into the future, ignoring all other 
information. We hope to beat this model with a LSTM recurrent network that 
considers current and recent values of the label and other features.
"""

class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

baseline = Baseline(label_index=column_indices['flights_ontime'])

baseline.compile(loss=MeanSquaredError(),
                 metrics=[MeanAbsoluteError(), MeanSquaredError(), MeanAbsolutePercentageError()])

baseline_results = baseline.evaluate(w1.val, verbose=0)

val_performance = {
   'baseline_Keras_evaluate': {
      'MSE': baseline_results[3],
      'MAE': baseline_results[1],
      'MAPE': baseline_results[2],
   }
}

# Sanity check comparing Keras and Sklearn baseline metrics
y_true = val_df['flights_ontime'].iloc[:-1]
y_pred = val_df['flights_ontime'].shift(-1).dropna()

val_performance['baseline_Sklearn'] = {
    'MSE': mean_squared_error(y_true, y_pred),
    'MAE': mean_absolute_error(y_true, y_pred),
    'MAPE': mean_absolute_percentage_error(y_true, y_pred),
}

print("Baseline validation performance:")
print(pd.DataFrame(val_performance).T.round(2))


# %% Baseline plot
w28.plot(baseline)

# %% Linear Model
"""
Predict one day into the future using a linear model. The model is a single layer with one neuron.
"""
linear = Sequential([Dense(units=1)])

print('Input shape:', w1.example[0].shape)
print('Output shape:', linear(w1.example[0]).shape)

# Design linear hypermodel
def build_linear_model(hp):
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-1, sampling='LOG', default=1e-2)
    
    model = Sequential()
    model.add(Dense(units=1, kernel_regularizer=L2(l2_reg)))
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])
    return model

# Design hyperband tuner
DenseLinear_tuner_HB = kt.Hyperband(build_linear_model,
                        objective='val_loss',
                        max_epochs=100,
                        factor=3,
                        directory='logs/flights_ontime/time_series/LinearDense',
                        project_name='hyperband_tuner')

# Get best hyperparameters for w1 model
!rm -rf logs/flights_ontime/time_series/LinearDense
early_stopping_HB = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
DenseLinear_tuner_HB.search(w1.train, 
             epochs=10, 
             validation_data=w1.val,
             callbacks=[early_stopping_HB])
best_w1_hps = DenseLinear_tuner_HB.get_best_hyperparameters(num_trials=1)[0]

# Build and train the best w1 model
DenseLinear = DenseLinear_tuner_HB.hypermodel.build(best_w1_hps)
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
history = DenseLinear.fit(w1.train, 
                    epochs=500, 
                    validation_data=w1.val, 
                    callbacks=[early_stopping],
                    verbose=0)

# Dense linear model performance
"""
I'm unable to get Sklearn's metrics to agree with those from Keras model.evaluate. I suspect 
the y_true and y_pred are mis-aligned. Consequently, I removed the 'flights_ontime' target 
from the preprocessor scaler, so that the MSE, MAE, and MAPE from the evaluate method are 
based on the raw data scale. This means we can compare Keras models for predicting 
flights_ontime to other models without having to inverse transform the predictions before 
calcluating metrics with Sklearn. I "think" the only downside is potentially slower model 
fits.
"""
val_performance['DenseLinear_Keras_evaluate'] = {
    'MSE': DenseLinear.evaluate(w1.val, verbose=0)[0],
    'MAE': DenseLinear.evaluate(w1.val, verbose=0)[1],
    'MAPE': DenseLinear.evaluate(w1.val, verbose=0)[2]
    }

y_true = np.concatenate([y for x, y in w1.val], axis=0).reshape(-1,1)
y_pred = DenseLinear.predict(w1.val).reshape(-1,1)


val_performance['DenseLinear_Sklearn'] = {
    'MSE': mean_squared_error(y_true, y_pred),
    'MAE': mean_absolute_error(y_true, y_pred),
    'MAPE': mean_absolute_percentage_error(y_true, y_pred)
    }

print("Validation set performance:")
print(pd.DataFrame(val_performance).T.round(2))
# %% Dense Linear Model plot
w28.plot(DenseLinear)

# %% Long Short Term Memory (LSTM) hypermodel
# Build LSTM Hypermodel
def build_lstm_model(hp):
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-1, sampling='LOG', default=1e-2)
    
    model = Sequential([tf.keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=L2(l2_reg)),
                        tf.keras.layers.Dense(units=1)])
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])
    return model

# %% tune_and_evaluate Class
def tune_and_evaluate(window, tuner_type='hyperband', epochs=100, patience=10, max_trials=100):
   # clear logs
    !rm -rf logs/flights_ontime/time_series/LSTM

    if tuner_type == 'random_search':
        LSTM_tuner = kt.RandomSearch(build_lstm_model,
                        objective='val_loss',
                        max_trials=max_trials,
                        directory='logs/flights_ontime/time_series/LSTM',
                        project_name='random_search_tuner')
    else:
        LSTM_tuner = kt.Hyperband(build_lstm_model,
                        objective='val_loss',
                        max_epochs=epochs,
                        factor=3,
                        directory='logs/flights_ontime/time_series/LSTM',
                        project_name='hyperband_tuner')
        
    # Get best hyperparameters
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    LSTM_tuner.search(window.train,
                        epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    
    best_hps = LSTM_tuner.get_best_hyperparameters(num_trials=1)[0]
    LSTM_model = LSTM_tuner.hypermodel.build(best_hps)
    history = LSTM_model.fit(window.train,
                            epochs=500,
                            validation_data=window.val,
                            callbacks=[early_stopping],
                            verbose=0)
    
    # LSTM model performance
    LSTM_Keras_evaluate = {
       'MSE': LSTM_model.evaluate(window.val, verbose=0)[0],
       'MAE': LSTM_model.evaluate(window.val, verbose=0)[1],
       'MAPE': LSTM_model.evaluate(window.val, verbose=0)[2]
       }
    
    y_true = np.concatenate([y for x, y in window.val], axis=0).reshape(-1,1)
    y_pred = LSTM_model.predict(window.val).reshape(-1,1)

    LSTM_Sklearn = {
        'MSE': mean_squared_error(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred)
        }
    
    return LSTM_model, LSTM_Keras_evaluate, LSTM_Sklearn

# %% Tune, evaluate, save LSTM models
os.makedirs('models/flights_ontime/TimeSeries', exist_ok=True)

LSTMw1, LSTMw1_Keras_evaluate, LSTMw1_Sklearn = tune_and_evaluate(w1)
LSTMw1.save('models/flights_ontime/TimeSeries/LSTMw1.keras')
val_performance['LSTMw1_Keras_evaluate'] = LSTMw1_Keras_evaluate
val_performance['LSTMw1_Sklearn'] = LSTMw1_Sklearn

LSTMw2, LSTMw2_Keras_evaluate, LSTMw2_Sklearn = tune_and_evaluate(w2)
LSTMw2.save('models/flights_ontime/TimeSeries/LSTMw2.keras')
val_performance['LSTMw2_Keras_evaluate'] = LSTMw2_Keras_evaluate
val_performance['LSTMw2_Sklearn'] = LSTMw2_Sklearn

LSTMw3, LSTMw3_Keras_evaluate, LSTMw3_Sklearn = tune_and_evaluate(w3)
LSTMw3.save('models/flights_ontime/TimeSeries/LSTMw3.keras')
val_performance['LSTMw3_Keras_evaluate'] = LSTMw3_Keras_evaluate
val_performance['LSTMw3_Sklearn'] = LSTMw3_Sklearn

LSTMw7, LSTMw7_Keras_evaluate, LSTMw7_Sklearn = tune_and_evaluate(w7)
LSTMw7.save('models/flights_ontime/TimeSeries/LSTMw7.keras')
val_performance['LSTMw7_Keras_evaluate'] = LSTMw7_Keras_evaluate
val_performance['LSTMw7_Sklearn'] = LSTMw7_Sklearn

LSTMw14, LSTMw14_Keras_evaluate, LSTMw14_Sklearn = tune_and_evaluate(w14)
LSTMw14.save('models/flights_ontime/TimeSeries/LSTMw14.keras')
val_performance['LSTMw14_Keras_evaluate'] = LSTMw14_Keras_evaluate
val_performance['LSTMw14_Sklearn'] = LSTMw14_Sklearn

LSTMw28, LSTMw28_Keras_evaluate, LSTMw28_Sklearn = tune_and_evaluate(w28)
LSTMw28.save('models/flights_ontime/TimeSeries/LSTMw28.keras')
val_performance['LSTMw28_Keras_evaluate'] = LSTMw28_Keras_evaluate
val_performance['LSTMw28_Sklearn'] = LSTMw28_Sklearn

print("Validation set performance:")
print(pd.DataFrame(val_performance).T.round(2))

# Save validation performance metrics
df = pd.DataFrame(val_performance).T
df['model_library'] = df.index
df['model_library'] = df['model_library'].str.replace('_evaluate', '')
df[['model', 'library']] = df['model_library'].str.split('_', expand=True)
df = df.drop(columns='model_library')
df = df[['model', 'library', 'MSE', 'MAE', 'MAPE']]
df = df.reset_index(drop=True)
df.to_csv('model_output/TimeSeries_results.csv')

# %%
df
# %% LSTM Model performance (OLD)
print("Validation set performance:")
print(df.round(2))





# # %% Build tuners
# # Delete logs directory
# !rm -rf logs/flights_ontime/time_series/LSTM

# # Hyperband tuner
# LSTM_tuner_HB = kt.Hyperband(build_lstm_model,
#                         objective='val_loss',
#                         max_epochs=100,
#                         factor=3,
#                         directory='logs/flights_ontime/time_series/LSTM',
#                         project_name='hyperband_tuner')

# # Random Search tuner
# LSTM_tuner_RS = kt.RandomSearch(build_lstm_model,
#                         objective='val_loss',
#                         max_trials=10,
#                         directory='logs/flights_ontime/time_series/LSTM',
#                         project_name='random_search_tuner')

# # Hyperband hyperparameter search using w2 data
# # early_stopping_HB = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# # LSTM_tuner_HB.search(w2.train,
# #                      epochs=10,
# #                      validation_data=wide_window.val,
# #                      callbacks=[early_stopping_HB])

# # Random hyperparameter search using w2 data
# early_stopping_RS = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
# LSTM_tuner_RS.search(w2.train,
#                      epochs=100,
#                      validation_data=w2.val,
#                      callbacks=[early_stopping_RS])

# # Get best hyperparameters for w2 model
# best_w2_hps = LSTM_tuner_RS.get_best_hyperparameters(num_trials=1)[0]
# LSTM_w2 = LSTM_tuner_RS.hypermodel.build(best_w2_hps)

# # Train the best w2 model
# early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
# history = LSTM_w2.fit(w2.train, 
#                     epochs=500, 
#                     validation_data=w2.val, 
#                     callbacks=[early_stopping],
#                     verbose=0)

# # LSTM_w2 model performance
# val_performance['LSTM_Keras_evaluate'] = {
#     'MSE': LSTM_w2.evaluate(w2.val, verbose=0)[0],
#     'MAE': LSTM_w2.evaluate(w2.val, verbose=0)[1],
#     'MAPE': LSTM_w2.evaluate(w2.val, verbose=0)[2]
#     }

# # Sanity check comparing Keras and Sklearn LSTM metrics
# y_true = np.concatenate([y for x, y in w2.val], axis=0).reshape(-1,1)
# y_pred = LSTM_w2.predict(w2.val).reshape(-1,1)

# val_performance['LSTM_Sklearn'] = {
#     'MSE': mean_squared_error(y_true, y_pred),
#     'MAE': mean_absolute_error(y_true, y_pred),
#     'MAPE': mean_absolute_percentage_error(y_true, y_pred)
#     }

# print("Validation set performance:")
# print(pd.DataFrame(val_performance).T.round(2))
# # %%
# w28.plot(LSTM_w2)
# # %%
# # %% Save model and performance metrics
# # Save model
# LSTM_w2.save('models/flights_ontime/LSTM_w2')

# # Save performance metrics
# val_performance_df = pd.DataFrame(val_performance).T
# val_performance_df.to_csv('models/flights_ontime/val_performance.csv')
# # %%
# # %% Load model and performance metrics
# # Load model
# LSTM_w2 = models.load_model('models/flights_ontime/LSTM_w2')
