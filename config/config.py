"""
Configuration file for MediOptima.
Centralizes constants and parameters to avoid hardcoding throughout the project.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
DATA_PATH = os.path.join(BASE_DIR, "data", "hospital_data.csv")

# Model configurations
FORECAST_DAYS = 7
DEFAULT_MODEL_TYPE = "Prophet" # Options: ARIMA, Prophet, LSTM

# Optimization Constants
# Ratios: 1 doctor per 20 patients, 1 nurse per 2 ICU patients
DOC_TO_PATIENT_RATIO = 20
NURSE_TO_ICU_RATIO = 2

# Assumed costs for linear programming (per shift/day)
STAFF_COST = {
    "doctor": 1000, 
    "nurse": 500
}

# Buffer constraints
EMERGENCY_BUFFER = 0.15 # 15% extra beds for emergency buffer
