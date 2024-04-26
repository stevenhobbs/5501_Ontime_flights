#%% LIBRARIES
import os
import shutil

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

import tensorflow as tf

from keras import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import L2
from keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError

import keras_tuner as kt

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

import warnings
warnings.filterwarnings('ignore')

# %% IMPORT DATA AND DEFINE COLUMN GROUPS
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
date_cols = ['covid', 'ordinal_date', 'year', 'month', 'day_of_month', 'day_of_week', 'season', 'holiday', 'halloween', 'xmas_eve', 'new_years_eve', 'jan_2', 'jan_3', 'day_before_easter', 'days_until_xmas', 'days_until_thanksgiving', 'days_until_july_4th', 'days_until_labor_day', 'days_until_memorial_day']

# Weather column groups
weather_cols = ['wx_temperature_max', 'wx_temperature_min', 'wx_apcp', 'wx_prate', 'wx_asnow', 'wx_frozr', 'wx_vis', 'wx_gust', 'wx_maxref', 'wx_cape', 'wx_lftx', 'wx_wind_speed', 'wx_wind_direction']
weather_cols_s2 = ['wx_temperature_max_s2', 'wx_temperature_min_s2', 'wx_apcp_s2', 'wx_prate_s2', 'wx_asnow_s2', 'wx_frozr_s2', 'wx_vis_s2', 'wx_gust_s2', 'wx_maxref_s2', 'wx_cape_s2', 'wx_lftx_s2', 'wx_wind_speed_s2', 'wx_wind_direction_s2']

# Lag column groups
lag_cols =  ['flights_cancel_lag_1', 'flights_cancel_lag_2', 'flights_cancel_lag_3', 'flights_cancel_lag_4', 'flights_cancel_lag_5', 'flights_cancel_lag_6', 'flights_cancel_lag_7',
             'flights_delay_lag_1', 'flights_delay_lag_2', 'flights_delay_lag_3', 'flights_delay_lag_4', 'flights_delay_lag_5', 'flights_delay_lag_6', 'flights_delay_lag_7',
             'flights_ontime_lag_1', 'flights_ontime_lag_2', 'flights_ontime_lag_3', 'flights_ontime_lag_4', 'flights_ontime_lag_5', 'flights_ontime_lag_6', 'flights_ontime_lag_7']
# %% Select columns for time series analysis
# Drop lag columns and date from data
ts_df = df[date_cols + flights_prediction_cols + weather_cols + weather_cols_s2]

print("Unique data types in LSTM_df", ts_df.dtypes.value_counts(), sep = '\n')


# Identify categorical and numeric columns in df
categorical_cols = ts_df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = ts_df.select_dtypes(include = ['float64', 'float32', 'int32', 'int64']).columns.tolist()
num_features = ts_df.shape[1]

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numeric columns: {numeric_cols}")
print(f"\nAll columns accounted for: {len(categorical_cols) + len(numeric_cols) == num_features}")

# %% SPLIT DATA SEQUENTIALLY 80-10-10
n = len(ts_df)
train_raw = ts_df[0:int(n*0.8)]
val_raw = ts_df[int(n*0.8):int(n*0.9)]
test_raw = ts_df[int(n*0.9):]

# print data shapes
print(f"Train data preprocessed shape: {train_raw.shape}")
print(f"Validation preprocessed data shape: {val_raw.shape}")
print(f"Test data preprocessed shape: {test_raw.shape}")

# %% PREPROCESS DATA FOR TIME SERIES
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
# %% WINDOW GENERATOR CLASS
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
               label_columns=None, batch_size=32):
               
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.batch_size = batch_size

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
  
  def plot(self, model="", plot_col='flights_ontime', max_subplots=3, filepath="", filename='default_plot.png'):
    """
    Given a model and a column to plot, the method plots the inputs, labels, and predictions

    Args:
    model: A trained model
    plot_col: The column to plot
    max_subplots: The maximum number of subplots to display

    Returns:
    """
    assert filepath != "", "Filepath to save the plot must be provided"

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
        if model:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)
        if n == 0:
            plt.legend()

        # Label axis
        plt.xlabel('Time [days]')

    
    # Make the destination directory if it does not exist
    if model: 
       directory = filepath + f"{model.name}/"
       model_name = model.name
    else: 
       directory = filepath
       model_name = "inputs_and_labels"

    # Save the plot
    print(f"Saving {model_name} plot to {directory}")
    os.makedirs(directory, exist_ok=True)
    plt.savefig(directory + filename)

    # Close the plot
    plt.close() 
    
  def make_dataset(self, data, batch_size, seed=42):
    data = np.array(data, dtype=np.float32)
    # Calculate the number of complete batches possible
    n = len(data)
    num_complete_batches = n // self.total_window_size
    max_data_length = num_complete_batches * batch_size * self.total_window_size
    data = data[:max_data_length]
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        seed = seed,
        batch_size=batch_size,)
    
    ds = ds.map(self.split_window)
    return ds
  
  @property
  def train(self):
    return self.make_dataset(self.train_df, self.batch_size)

  @property
  def val(self):
    return self.make_dataset(self.val_df, self.batch_size)

  @property
  def test(self):
     return self.make_dataset(self.test_df, self.batch_size)

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
  
#%% INSTANTIATE WINDOW GENERATORS
"""
Baseline and deep neural network (DNN) models always predict one time step (one day)
into the future, based on a single day of information (the current day). Technically, these are
not time series models, because they use only one time step for prediction. For baseline and DNN
models, the choice of input width only serves as another batch size variable and determines the width 
of the plot window, but does not change the amount of information used to make each prediction 
(still just 1 day). For baseline and DNN models, we'll use a width of 14 for the inputs and labels. 

For LSTM models, the input width is the number of days used to predict the next day, and the label width is the number of days predicted. The shift is
the number of days between the last day of the input window and the first day of the label window. 
"""

# 1 and 16 day window for baseline and DNN models
w1_dnn = WindowGenerator(input_width=1, label_width=1, shift=1,
                          label_columns=['flights_ontime'])

w14_dnn = WindowGenerator(input_width=14, label_width=14, shift=1,
                          label_columns=['flights_ontime'])


# Windows for LSTM models 

widths = [1, 2, 4, 8, 16, 32]

windows = {}

for width in widths:
    window = f"w{width}"
    windows[window] = WindowGenerator(input_width=width, label_width=1, shift=1,
                                     label_columns=['flights_ontime'])
    
# Unpack the windows
w1, w2, w4, w8, w16, w32 = windows.values()

# %% WINDOW SPLIT EXAMPLES
print('Window shape = (batch size, time steps, features)')

# 14 day baseline and dnn window example
example_batch = tf.stack([np.array(train_df[:w14_dnn.total_window_size]),
                            np.array(train_df[100:100+w14_dnn.total_window_size]),
                            np.array(train_df[200:200+w14_dnn.total_window_size])
                            ])

example_inputs, example_labels = w14_dnn.split_window(example_batch)

print(f'\n14 day DNN window shape: {example_batch.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

# Eight day time series window example
example_batch = tf.stack([np.array(train_df[:w8.total_window_size]),
                           np.array(train_df[100:100+w8.total_window_size]),
                           np.array(train_df[200:200+w8.total_window_size])])

example_inputs, example_labels = w8.split_window(example_batch)

print(f'\n8 day Time Series window shape: {example_batch.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')

# %% EXAMPLE PLOTS OF INPUTS AND LABELS 
output_dir = "output/flights_ontime/TF_Time_Series/"
w14_dnn.plot(filepath=output_dir, filename = "w14_dnn_inputs_and_labels_example_plot.png")
w8.plot(filepath=output_dir, filename = "w8_inputs_and_labels_example_plot.png")

# %% BASELINE FORECASTING
"""
The `Baseline` class inherits from `keras.Model` and uses the current value of a 
label to predict a label one step (one day) into the future, ignoring all other 
information. We hope to beat this model with a LSTM recurrent network that 
considers current and recent values of the label and other features.
"""

class BaselineNaive(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

BaselineNaive = BaselineNaive(label_index=column_indices['flights_ontime'])
BaselineNaive.compile(loss=MeanSquaredError(),
                 metrics=[MeanAbsoluteError(), 
                          MeanSquaredError(), 
                          MeanAbsolutePercentageError()])

BaselineNaive_results = BaselineNaive.evaluate(w1.val, verbose=0)

val_performance = {
   'BaselineNaive_Keras_evaluate': {
      'MSE': BaselineNaive_results[3],
      'MAE': BaselineNaive_results[1],
      'MAPE': BaselineNaive_results[2]
   }
}

# Sanity check comparing Keras and Sklearn BaselineNaive metrics
y_true = val_df['flights_ontime'].iloc[:-1]
y_pred = val_df['flights_ontime'].shift(-1).dropna()

val_performance['BaselineNaive_Sklearn'] = {
    'MSE': mean_squared_error(y_true, y_pred),
    'MAE': mean_absolute_error(y_true, y_pred),
    'MAPE': mean_absolute_percentage_error(y_true, y_pred)
}

print("BaselineNaive validation performance:")
print(pd.DataFrame(val_performance).T.round(2))


# %% BASELINE NAIVE PLOT
w14_dnn.plot(model=BaselineNaive, filepath=output_dir, filename='w14_dnn_baseline_naive_plot.png')

# %% DEFINE TUNE_AND_EVALUATE FUNCTION

"""
I'm unable to get Sklearn's metrics to agree with Keras model.evaluate when modeling with
a scaler-transformed target. I suspect the windowing function is causing y_true and the 
inverse-transformed y_pred are mis-aligned. Consequently, I removed the 'flights_ontime' target 
from the preprocessor scaler, so that the MSE, MAE, and MAPE from the evaluate method are 
based on the raw data scale, without inverse-transforming the predictions. I "think" the only 
downside is potentially slower model fits.
"""
def tune_and_evaluate(window, model_name, hypermodel_func, val_performance_dict, 
                      tuner_type='hyperband', epochs=100, patience=10, max_trials=100, verbose = 0):
   # clear logs
   logs_dir = f"logs/flights_ontime/time_series/{hypermodel_func}"
   if os.path.exists(logs_dir):
      shutil.rmtree(logs_dir)
   os.makedirs(logs_dir)

   # Function to create hypermodel
   def model_builder(hp):
      return hypermodel_func(hp, model_name)

   # Define tuner
   if tuner_type == 'random_search':
      hypermodel_tuner = kt.RandomSearch(model_builder,
                                         objective='val_loss',
                                         max_trials=max_trials,
                                         directory=f'logs/flights_ontime/time_series/{hypermodel_func}',
                                         project_name='random_search_tuner')
    
   else:
    hypermodel_tuner = kt.Hyperband(model_builder,
                                    objective='val_loss',
                                    max_epochs=epochs,
                                    factor=3,
                                    directory=f'logs/flights_ontime/time_series/{hypermodel_func}',
                                    project_name='hyperband_tuner')
    
   # Get best hyperparameters
   early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
   hypermodel_tuner.search(window.train,
                           epochs=epochs,
                           validation_data=window.val,
                           callbacks=[early_stopping],
                           verbose=verbose)
   best_hps = hypermodel_tuner.get_best_hyperparameters(num_trials=1)[0]

   # Build and train the model with the best hyperparameters
   early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
   Model = hypermodel_tuner.hypermodel.build(best_hps)
   history = Model.fit(window.train,
                       epochs=500,
                       validation_data=window.val,
                       callbacks=[early_stopping],
                       verbose=verbose)
     
   # Validation set performance using Keras evaluate method
   val_performance_dict[f'{model_name}_Keras_evaluate'] = {
       'MSE': Model.evaluate(window.val, verbose=verbose)[0],
       'MAE': Model.evaluate(window.val, verbose=verbose)[1],
       'MAPE': Model.evaluate(window.val, verbose=verbose)[2]
       }
   
   # Validation set performance using Sklearn metrics
   y_true = np.concatenate([y for x, y in window.val], axis=0).reshape(-1,1)
   y_pred = Model.predict(window.val).reshape(-1,1)
   val_performance_dict[f'{model_name}_Sklearn'] = {
      'MSE': mean_squared_error(y_true, y_pred),
      'MAE': mean_absolute_error(y_true, y_pred),
      'MAPE': mean_absolute_percentage_error(y_true, y_pred)
      }
   
   # Save the model
   models_dir = f'models/flights_ontime/TF_time_series/{model_name}'
   os.makedirs(models_dir, exist_ok=True)
   Model.save(models_dir + f'/{model_name}.keras')
   
   return Model, val_performance_dict

# %% DNN (DENSE NEURAL NETWORK) - LINEAR HYPERMODEL

def build_TimeSeriesDNNLinear(hp, model_name='default_model_name'):
    input_neurons = hp.Int('neurons', min_value=1, max_value=64, step=1, default=1)
    hidden_neurons = hp.Int('hidden_neurons', min_value=1, max_value=64, step=1, default=1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1, default=0.5)
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-1, sampling='LOG', default=1e-2)
    
    model = Sequential(name=model_name)
    # Input layer
    model.add(Dense(units=input_neurons, 
                    kernel_regularizer=L2(l2_reg)))
    model.add(Dropout(dropout_rate))

    # Hidden layer
    model.add(Dense(units=hidden_neurons,
                    kernel_regularizer=L2(l2_reg)))
    model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])
    return model

# %% DNN LINEAR - TUNE AND VALIDATE 
dnn_test_settings = {'hypermodel_func' : build_TimeSeriesDNNLinear,
                      'val_performance_dict': val_performance,
                      'epochs' : 2,
                      'patience' : 1,
                      'max_trials' : 2}

dnn_run_settings = {'hypermodel_func' : build_TimeSeriesDNNLinear,
                    'val_performance_dict': val_performance,
                    'epochs' : 500, 
                    'patience' : 20,
                    'max_trials' : 500}

# settings = dnn_test_settings
settings = dnn_run_settings

# Tune and evalutethe 1 and 14-day DNN Linear models
DNNLinearW1, val_performance = tune_and_evaluate(window=w1_dnn,
                                                 model_name='DNNLinearW1',
                                                 **settings,)

DNNLinearW14, val_performance = tune_and_evaluate(window=w14_dnn,
                                                  model_name='DNNLinearW14',
                                                  **settings,)


# %% DNN LINEAR - MODEL SUMMARY AND VALIDATION PERFORMANCE
# Hyperparameter Summary
print("DNN Linear 1-day window", DNNLinearW1.summary())
print("DNN Linear 14-day window", DNNLinearW14.summary())

# Validation Performance
print("Validation set performance:")
print(pd.DataFrame(val_performance).T.round(2))

# %% DNN NON-LINEAR - HYPERMODEL (RELU ACTIVATION)
def build_TimeSeriesDNNNonLinear(hp, model_name='default_model_name'):
    input_neurons = hp.Int('neurons', min_value=1, max_value=64, step=1, default=1)
    hidden_neurons = hp.Int('hidden_neurons', min_value=1, max_value=64, step=1, default=1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1, default=0.5)
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-1, sampling='LOG', default=1e-2)
    
    model = Sequential(name=model_name)
    model.add(Dense(units=input_neurons, activation='relu', 
                    kernel_regularizer=L2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=hidden_neurons, activation='relu',
                    kernel_regularizer=L2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])
    return model

# %% DNN NON-LINEAR - TUNE AND VALIDATE

dnn_test_settings = {'hypermodel_func' : build_TimeSeriesDNNNonLinear,
                      'val_performance_dict': val_performance,
                      'epochs' : 2,
                      'patience' : 1,
                      'max_trials' : 2}

dnn_run_settings = {'hypermodel_func' : build_TimeSeriesDNNNonLinear,
                    'val_performance_dict': val_performance,
                    'epochs' : 500, 
                    'patience' : 20,
                    'max_trials' : 500}

settings = dnn_run_settings

# Tune and evaluate the 1 and 14-day DNN Non-Linear models
DNNNonLinearW1, val_performance = tune_and_evaluate(window=w1_dnn,
                                                   model_name='DNNNonLinearW1',
                                                   **settings)

DNNNonLinearW14, val_performance = tune_and_evaluate(window=w14_dnn,
                                                      model_name='DNNNonLinearW14',
                                                      **settings)

# %% DNN NON-LINEAR MODEL SUMMARY AND VALIDATION PERFORMANCE
# Hyperparameter Summary
print("DNN Non-Linear 1-day window", DNNNonLinearW1.summary())
print("DNN Non-Linear 14-day window", DNNNonLinearW14.summary())

# Validation Performance
print("Validation set performance:")
print(pd.DataFrame(val_performance).T.round(2))



# %% LSTM - HYPERMODEL
# Build LSTM Hypermodel
def build_TimeSeriesLSTM(hp, model_name='default_model_name'):
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-1, sampling='LOG', default=1e-2)
    num_layers = hp.Int('num_layers', min_value=1, max_value=3, default=1)  # Hyperparameter for the number of LSTM layers
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, default=0.25)  # Hyperparameter for dropout rate
    
    model = Sequential()
    for i in range(num_layers):
        # Add LSTM layers
        return_sequences = True if i < num_layers - 1 else False  # Only the last LSTM layer has return_sequences=False
        units = hp.Int(f'units_{i+1}', min_value=32, max_value=256, step=32)  # Hyperparameter for units in each LSTM layer
        model.add(LSTM(units=units,
                       return_sequences=return_sequences,
                       kernel_regularizer=L2(l2_reg)))
        model.add(Dropout(rate=dropout_rate))  # Adding dropout after each LSTM layer
    
    # Output layer
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error',
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])
    return model

# ORIGINAL MODEL WITH 1 LSTM LAYER
# def build_TimeSeriesLSTM(hp, model_name='default_model_name'):
#     learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
#     l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-1, sampling='LOG', default=1e-2)
    
#     model = Sequential([LSTM(units=64, 
#                              return_sequences=False, 
#                              kernel_regularizer=L2(l2_reg)),
#                         Dense(units=1)])
#     model.compile(loss='mean_squared_error', 
#                   optimizer=Adam(learning_rate=learning_rate),
#                   metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])
#     return model



# %% LSTM -TUNE AND VALIDATE

# Tune and evaluate configurations for LSTM models
test_settings = {'hypermodel_func' : build_TimeSeriesLSTM,
                 'val_performance_dict': val_performance,
                 'epochs' : 2,
                 'patience' : 1,
                 'max_trials' : 2}

run_settings = {'hypermodel_func' : build_TimeSeriesLSTM,
                'val_performance_dict': val_performance,
                'epochs' : 500, 
                'patience' : 20,
                'max_trials' : 500}

lstm_windows = {1: w1, 
                2: w2, 
                4: w4, 
                8: w8, 
                16: w16}

settings = run_settings

models_dict = {}

for days, window in lstm_windows.items():
    print(f"\nTune and evaluate LSTM {days}-day model")
    model_name = f"TimeSeriesLSTMW{days}"
    model, val_performance = tune_and_evaluate(window=window,
                                               model_name=model_name,
                                               **settings)
    models_dict[model_name] = model
    print("Time Series LSTM", days, "day window\n") 
    model.summary()
  

# %% LSTM - PERFORMANCE
print("Validation set performance:")
print(pd.DataFrame(val_performance).T.round(2))

# %% Save validation performance to a CSV file
output_dir = "model_output/daily/LSTM"
os.makedirs(output_dir, exist_ok=True)
val_performance_df = pd.DataFrame(val_performance).T.round(2)
val_performance_df.to_csv(output_dir + "/TFlowTimeSeries_val.csv")


