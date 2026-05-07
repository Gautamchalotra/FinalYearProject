"""
Script to generate realistic synthetic hospital data for training and testing.
Simulates daily patient inflow, ICU limits, available beds, and staff constraints.
Now supports multi-year generation (4 years) and multiple hospital branches!
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_hospital_node(hospital_id: str, days: int, end_date: datetime, base_patients: int, rand_seed: int):
    """
    Generates chronological synthetic data for a single hospital.
    """
    np.random.seed(rand_seed) # Ensure reproducibility per hospital
    
    dates = [end_date - timedelta(days=x) for x in range(days)]
    dates.reverse() # Sort chronologically
    
    t = np.arange(days)
    
    # 1. Trend (gradual increase of patients over the 4 years)
    trend = t * 0.03
    
    # 2. Seasonality (Weekly + Yearly Winter spikes)
    # Weekly patterns: Weekends (5,6) usually see drops in elective/regular admits, but ER stays same.
    days_of_week = pd.Series(dates).dt.dayofweek
    weekly_seasonality = np.where(days_of_week >= 5, -20, 15) 
    
    # Yearly pattern: Flu season peaks around January (Day 30), drops in Summer (Day 200)
    day_of_year = pd.Series(dates).dt.dayofyear
    yearly_seasonality = 35 * np.cos(2 * np.pi * (day_of_year - 30) / 365.25)
    
    # 3. Noise (Random daily variation)
    noise = np.random.normal(0, 15, days)
    
    # Compute base patients
    patients = base_patients + trend + weekly_seasonality + yearly_seasonality + noise
    
    # 4. Sudden Spikes & Rare Pandemic Events
    # Pandemic scenario (random large bell curve spanning 2 months)
    pandemic_peak = np.zeros(days)
    pan_start = int(days * np.random.uniform(0.3, 0.7))
    pan_length = 60 # 60 days
    if pan_start + pan_length < days:
        x = np.linspace(-3, 3, pan_length)
        # Bell curve peaking at 50-100 extra patients
        bell = np.random.randint(50, 100) * np.exp(-0.5 * x**2)
        pandemic_peak[pan_start:pan_start+pan_length] = bell
        
    patients += pandemic_peak
    
    # Random sudden extreme emergencies (accidents, local outbreaks) - 2% of the days
    emergency_spikes = np.zeros(days)
    spike_indices = np.random.choice(range(days), size=int(days*0.02), replace=False)
    for idx in spike_indices:
        emergency_spikes[idx] = np.random.randint(40, 120)
        
    patients += emergency_spikes
    
    # Floor to absolute minimum to avoid negative or unrealistic drops
    patients = np.maximum(50, patients).astype(int)
    
    # 5. Corelate Operational Metrics
    # ICU is strictly related to patients, usually 10-20%
    icu_patients = (patients * np.random.uniform(0.1, 0.2, size=days)).astype(int)
    
    # ER is driven heavily by the emergency spikes + baseline walk-ins
    base_er = patients * np.random.uniform(0.2, 0.35, size=days)
    emergency_cases = (base_er + emergency_spikes * 0.9).astype(int)
    
    # Discharges lag slightly/are slightly less than admits on growth days to maintain hospitalized counts
    discharges = (patients * np.random.uniform(0.85, 0.98, size=days)).astype(int)
    
    # 6. Establish Edge Cases (Shortages)
    # Available beds fluctuate slightly based on maintenance, but mostly stable around standard capacity
    standard_capacity = base_patients + 150
    available_beds = np.full(days, standard_capacity) - np.random.randint(0, 10, days)
    
    # Staff constraints
    # Under normal conditions, proportional. But let's add some staffing dips (holidays, strikes)
    doctors = (patients / 20).astype(int) + np.random.randint(-1, 3, days)
    nurses = (patients / 10).astype(int) + np.random.randint(-3, 5, days)
    
    # Create severe staff shortages for Anomaly Detection edge cases
    shortage_indices = np.random.choice(range(days), size=int(days*0.01), replace=False)
    for idx in shortage_indices:
        nurses[idx] = max(5, nurses[idx] - np.random.randint(15, 25))
        doctors[idx] = max(2, doctors[idx] - np.random.randint(5, 10))

    # Cap mins
    doctors = np.maximum(2, doctors)
    nurses = np.maximum(5, nurses)
    
    # Available ICU beds
    available_icu_beds = np.full(days, int(standard_capacity * 0.15))
    
    # 7. Medicine Usage
    antibiotics = (patients * np.random.uniform(1.2, 1.8, size=days)).astype(int)
    icu_drugs = (icu_patients * np.random.uniform(2.5, 3.5, size=days)).astype(int)
    painkillers = (patients * np.random.uniform(0.5, 0.8, size=days) + emergency_cases * np.random.uniform(1.0, 1.5, size=days)).astype(int)
    total_medicine = antibiotics + icu_drugs + painkillers
    
    return pd.DataFrame({
        'hospital_id': hospital_id,
        'date': dates,
        'patients': patients,
        'icu_patients': icu_patients,
        'emergency_cases': emergency_cases,
        'discharges': discharges,
        'available_beds': available_beds,
        'available_icu_beds': available_icu_beds,
        'doctors': doctors,
        'nurses': nurses,
        'antibiotics': antibiotics,
        'painkillers': painkillers,
        'icu_drugs': icu_drugs,
        'total_medicine': total_medicine
    })

def generate_multi_hospital_data(years=4, output_file="data/hospital_data.csv"):
    """
    Generates data for multiple hospital locations and amalgamates them.
    """
    try:
        days = 365 * years # Roughly 1460 rows per hospital
        end_date = datetime.today()
        
        # Configuration for our distinct hospitals
        hospital_configs = [
            {"id": "HOSP-A", "base": 250, "seed": 42},    # Large metropolitan hospital
            {"id": "HOSP-B", "base": 120, "seed": 101}    # Medium suburban hospital
        ]
        
        logger.info(f"Generating {years} years of daily data for {len(hospital_configs)} hospitals...")
        
        all_data = []
        for conf in hospital_configs:
            df_node = generate_hospital_node(conf['id'], days, end_date, conf['base'], conf['seed'])
            all_data.append(df_node)
            
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure target directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save payload
        final_df.to_csv(output_file, index=False)
        logger.info(f"Successfully generated realistic dataset! Total Rows: {len(final_df)} | Output: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to generate hospital data: {str(e)}")

if __name__ == "__main__":
    generate_multi_hospital_data()
