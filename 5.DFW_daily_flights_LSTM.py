#%% LIBRARIES
import os
import shutil

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

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
# %% Select columns for LSTM modeling
# Drop lag columns and date from data
LSTM_df = df[date_cols + flights_prediction_cols + weather_cols + weather_cols_s2]

print("Unique data types in LSTM_df", LSTM_df.dtypes.value_counts(), sep = '\n')


# Identify categorical and numeric columns in df
categorical_cols = LSTM_df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = LSTM_df.select_dtypes(include = ['float64', 'float32', 'int32', 'int64']).columns.tolist()
num_features = LSTM_df.shape[1]

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numeric columns: {numeric_cols}")
print(f"\nAll columns accounted for: {len(categorical_cols) + len(numeric_cols) == num_features}")

# %% SPLIT DATA SEQUENTIALLY 80-10-10
n = len(LSTM_df)
train_raw = LSTM_df[0:int(n*0.8)]
val_raw = LSTM_df[int(n*0.8):int(n*0.9)]
test_raw = LSTM_df[int(n*0.9):]

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
# %% DEFINE WINDOW GENERATOR CLASS
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

    
    # Make the destination directory if it does not exist
    if model:
       print(f"Model name: {model.name}")
       directory = filepath + f"{model.name}/"
       print(f"Saving plot to {directory}")
    else:
       print("Model name: None")
       directory = filepath
       print(f"Saving plot to {directory}")
    
    os.makedirs(directory, exist_ok=True)

    plt.xlabel('Time [days]')
    plt.savefig(directory + filename)
    plt.close() # Close the plot to free up memory
    
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
  
#%% INSTANTIATE WINDOW GENERATORS
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
# %% DEMONSTRATE SPLIT_WINDOW METHOD
example_batch = tf.stack([np.array(train_df[:w7.total_window_size]),
                           np.array(train_df[100:100+w7.total_window_size]),
                           np.array(train_df[200:200+w7.total_window_size])])

example_inputs, example_labels = w7.split_window(example_batch)

print('Window shapes are: (batch size, time steps, features)')
print(f'w7 shape: {example_batch.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')
# %% EXAMPLE PLOT OF INPUTS AND LABELS FOR 7-DAY WINDOW
output_dir = "output/flights_ontime/TF_Time_Series/"
w7.plot(filepath=output_dir, filename = "w7_inputs_and_labels_example_plot.png")

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
                 metrics=[MeanAbsoluteError(), MeanSquaredError(), MeanAbsolutePercentageError()])

BaselineNaive_results = BaselineNaive.evaluate(w1.val, verbose=0)

val_performance = {
   'BaselineNaive_Keras_evaluate': {
      'MSE': BaselineNaive_results[3],
      'MAE': BaselineNaive_results[1],
      'MAPE': BaselineNaive_results[2],
   }
}

# Sanity check comparing Keras and Sklearn BaselineNaive metrics
y_true = val_df['flights_ontime'].iloc[:-1]
y_pred = val_df['flights_ontime'].shift(-1).dropna()

val_performance['BaselineNaive_Sklearn'] = {
    'MSE': mean_squared_error(y_true, y_pred),
    'MAE': mean_absolute_error(y_true, y_pred),
    'MAPE': mean_absolute_percentage_error(y_true, y_pred),
}

print("BaselineNaive validation performance:")
print(pd.DataFrame(val_performance).T.round(2))


# %% BASELINE PLOT
# w7.plot(model=baseline, filepath=output_dir, filename='w7_baseline_plot.png')
w28.plot(model=BaselineNaive, filepath=output_dir, filename='w28_baseline_naive_plot.png')

# %% DEFINE TUNE_AND_EVALUATE FUNCTION

"""
I'm unable to get Sklearn's metrics to agree with Keras model.evaluate when modeling with
a scaler-transformed target. I suspect the windowing function is causing y_true and the 
inverse-transformed y_pred are mis-aligned. Consequently, I removed the 'flights_ontime' target 
from the preprocessor scaler, so that the MSE, MAE, and MAPE from the evaluate method are 
based on the raw data scale, without inverse-transforming the predictions. I "think" the only 
downside is potentially slower model fits.
"""
def tune_and_evaluate(window, model_name, hypermodel_func, val_performance_dict, tuner_type='hyperband', epochs=100, patience=10, max_trials=100):
   # clear logs
   logs_dir = f"logs/flights_ontime/time_series/{hypermodel_func}"
   if os.path.exists(logs_dir):
      shutil.rmtree(logs_dir)
      os.makedirs(logs_dir)

   # Define tuner
   if tuner_type == 'random_search':
      hypermodel_tuner = kt.RandomSearch(lambda hp: hypermodel_func(hp, model_name),
                                         objective='val_loss',
                                         max_trials=max_trials,
                                         directory=f'logs/flights_ontime/time_series/{hypermodel_func}',
                                         project_name='random_search_tuner')
    
   else:
    hypermodel_tuner = kt.Hyperband(lambda hp: hypermodel_func(hp, model_name),
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
                           verbose=0)
   best_hps = hypermodel_tuner.get_best_hyperparameters(num_trials=1)[0]

   # Build and train the model with the best hyperparameters
   early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
   Model = hypermodel_tuner.hypermodel.build(best_hps)
   history = Model.fit(window.train,
                       epochs=500,
                       validation_data=window.val,
                       callbacks=[early_stopping],
                       verbose=0)
     
   # Validation set performance using Keras evaluate method
   val_performance_dict[f'{model_name}_Keras_evaluate'] = {
       'MSE': Model.evaluate(window.val, verbose=0)[0],
       'MAE': Model.evaluate(window.val, verbose=0)[1],
       'MAPE': Model.evaluate(window.val, verbose=0)[2]
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
   models_dir = f'models/flights_ontime/{model_name}'
   os.makedirs(models_dir, exist_ok=True)
   Model.save(models_dir + f'/{model_name}.keras')
   
   return Model, val_performance_dict

# %% DENSE NEURAL NETWORK (DNN) LINEAR HYPERMODEL

def build_TimeSeriesDNNLinear(hp, model_name='default_model_name'):
    input_neurons = hp.Int('neurons', min_value=1, max_value=64, step=1, default=1)
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1, default=0.5)
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-1, sampling='LOG', default=1e-2)
    
    model = Sequential(name=model_name)
    model.add(Dense(units=input_neurons, kernel_regularizer=L2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])
    return model

# %% TUNE AND VALIDATE DNN LINEAR MODELS

# One day window
TimeSeriesDNNLinearW1, val_performance = tune_and_evaluate(window=w1, 
                                                           model_name='TimeSeriesDNNLinearW1', 
                                                           hypermodel_func=build_TimeSeriesDNNLinear, 
                                                           val_performance_dict=val_performance)

# Two day window
TimeSeriesDNNLinearW2, val_performance = tune_and_evaluate(window=w2, 
                                                           model_name='TimeSeriesDNNLinearW2', 
                                                           hypermodel_func=build_TimeSeriesDNNLinear, 
                                                           val_performance_dict=val_performance)

# Three day window
TimeSeriesDNNLinearW3, val_performance = tune_and_evaluate(window=w3,
                                                            model_name='TimeSeriesDNNLinearW3',
                                                            hypermodel_func=build_TimeSeriesDNNLinear,
                                                            val_performance_dict=val_performance)

# Seven day window
TimeSeriesDNNLinearW7, val_performance = tune_and_evaluate(window=w7,
                                                            model_name='TimeSeriesDNNLinearW7',
                                                            hypermodel_func=build_TimeSeriesDNNLinear,
                                                            val_performance_dict=val_performance)


# %% Print Model Summaries and Validation Performance
# Model Hyperparameters
print("Time Series DNN Linear 1-day window", TimeSeriesDNNLinearW1.summary())
print("Time Series DNN Linear 2-day window", TimeSeriesDNNLinearW2.summary())
print("Time Series DNN Linear 3-day window", TimeSeriesDNNLinearW3.summary())
print("Time Series DNN Linear 7-day window", TimeSeriesDNNLinearW7.summary())

# Validation Performance
print("Validation set performance:")
print(pd.DataFrame(val_performance).T.round(2))

# %% Time Series Long Short Term Memory (LSTM) hypermodel
# Build LSTM Hypermodel
def build_TimeSeriesLSTM(hp, model_name='default_model_name'):
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG', default=1e-3)
    l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-1, sampling='LOG', default=1e-2)
    
    model = Sequential([LSTM(units=64, 
                             return_sequences=True, 
                             kernel_regularizer=L2(l2_reg)),
                        Dense(units=1)])
    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(learning_rate=learning_rate),
                  metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError()])
    return model

LSTM_hypermodel = build_TimeSeriesLSTM(kt.HyperParameters())
LSTM_hypermodel.summary()


# %% Tune and validate LSLTM models
# One day window
TimeSeriesLSTMW1, val_performance = tune_and_evaluate(window=w1,
                                                      model_name='TimeSeriesLSTMW1',
                                                      hypermodel_func=build_TimeSeriesLSTM,
                                                      val_performance_dict=val_performance)

# Two day window
TimeSeriesLSTMW2, val_performance = tune_and_evaluate(window=w2,
                                                      model_name='TimeSeriesLSTMW2',
                                                      hypermodel_func=build_TimeSeriesLSTM,
                                                      val_performance_dict=val_performance)

# Three day window
TimeSeriesLSTMW3, val_performance = tune_and_evaluate(window=w3,
                                                      model_name='TimeSeriesLSTMW3',
                                                      hypermodel_func=build_TimeSeriesLSTM,
                                                      val_performance_dict=val_performance)

# Seven day window
TimeSeriesLSTMW7, val_performance = tune_and_evaluate(window=w7,
                                                      model_name='TimeSeriesLSTMW7',
                                                      hypermodel_func=build_TimeSeriesLSTM,
                                                      val_performance_dict=val_performance)

# Fourteen day window 
TimeSeriesLSTMW14, val_performance = tune_and_evaluate(window=w14,
                                                       model_name='TimeSeriesLSTMW14',
                                                       hypermodel_func=build_TimeSeriesLSTM,
                                                       val_performance_dict=val_performance)

# Twenty-eight day window
TimeSeriesLSTMW28, val_performance = tune_and_evaluate(window=w28,
                                                       model_name='TimeSeriesLSTMW28',
                                                       hypermodel_func=build_TimeSeriesLSTM,
                                                       val_performance_dict=val_performance)

# %% Print Model Summaries and Validation Performance
# Model Hyperparameters
print("Time Series LSTM 1-day window", TimeSeriesLSTMW1.summary())
print("Time Series LSTM 2-day window", TimeSeriesLSTMW2.summary())
print("Time Series LSTM 3-day window", TimeSeriesLSTMW3.summary())
print("Time Series LSTM 7-day window", TimeSeriesLSTMW7.summary())
print("Time Series LSTM 14-day window", TimeSeriesLSTMW14.summary())
print("Time Series LSTM 28-day window", TimeSeriesLSTMW28.summary())

# Validation Performance
print("Validation set performance:")
print(pd.DataFrame(val_performance).T.round(2))

# %% Save validation performance to a CSV file
output_dir = "model_output/daily/LSTM"
os.makedirs(output_dir, exist_ok=True)
val_performance_df = pd.DataFrame(val_performance).T.round(2)
val_performance_df.to_csv(output_dir + "/LSTM_forecasts_val.csv")


