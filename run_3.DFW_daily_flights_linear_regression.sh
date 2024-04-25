#!/bin/bash
#SBATCH --job-name=LR_script
#SBATCH --output=kestrel_LR_results_%j.txt
#SBATCH --mem=32GB             
#SBATCH --time=01:00:00
#SBATCH --account=athena2
#SBATCH --cpus-per-task=8    

# navigate to project directory
cd ~/dfw/predicting_flight_traffic

# activate python environment and run script
pipenv run python3 3.DFW_daily_flights_linear_regression.py