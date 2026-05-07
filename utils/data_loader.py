"""
Data loader utility for the MediOptima application.
Handles reading and basic preprocessing of the generated dataset.
"""

import pandas as pd
import logging

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads hospital data from a CSV file and performs basic preprocessing.
    
    Args:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: A cleaned and parsed DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        # Parse date
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Fill missing values if any
        if df.isnull().values.any():
            logger.warning("Missing values detected in dataset. Forward filling...")
            df = df.ffill().bfill()
            
        logger.info(f"Successfully loaded data. Shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}. Please run generate_data.py first.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading data: {str(e)}")
        raise

def get_recent_metrics(df: pd.DataFrame) -> dict:
    """
    Extracts the most recent day's metrics for dashboard display.
    
    Args:
        df (pd.DataFrame): The preprocessed hospital historical data.
        
    Returns:
        dict: Dictionary containing the latest metrics.
    """
    latest = df.iloc[-1]
    return {
        "date": latest["date"],
        "patients": int(latest["patients"]),
        "icu_patients": int(latest["icu_patients"]),
        "emergency_cases": int(latest["emergency_cases"]),
        "available_beds": int(latest["available_beds"]),
        "available_icu_beds": int(latest["available_icu_beds"]),
        "doctors": int(latest["doctors"]),
        "nurses": int(latest["nurses"])
    }
