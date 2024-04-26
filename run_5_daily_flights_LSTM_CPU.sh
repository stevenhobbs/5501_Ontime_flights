#!/bin/bash
#SBATCH --job-name=LSTM_script
#SBATCH --output=kestrel_lstm_results_%j.txt
# #SBATCH --partition=standard   # Use a standard CPU partition, adjust the name as needed
#SBATCH --mem=32GB             
#SBATCH --time=02:00:00
#SBATCH --account=athena2
#SBATCH --cpus-per-task=8    

# navigate to project directory
cd ~/dfw/predicting_flight_traffic

# activate python environment and run script
pipenv run python3 5.DFW_daily_flights_LSTM.py
