#!/bin/bash
# #SBATCH --job-name=LSTM_script
#SBATCH --job-name=TensorFlowTest
# #SBATCH --output=kestrel_lstm_results_%j.txt
#SBATCH --output=kestrel_tensorflow_results_%j.txt
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --account=athena2

module load cuda/12.3

# navigate to project directory
cd ~/dfw/predicting_flight_traffic

# activate python environment and run script
# pipenv run python3 5.DFW_daily_flights_LSTM.py
pipenv run python3 TensorFlowTest.py