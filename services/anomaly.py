"""
Anomaly detection using Z-score and Isolation Forest to detect patient surges.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy import stats
import logging

logger = logging.getLogger(__name__)

def detect_anomalies(df: pd.DataFrame, target_col='patients') -> dict:
    """
    Detects recent anomalies (surges) in patient inflows.
    
    Args:
        df (pd.DataFrame): Historical data.
        target_col (str): Column to analyze.
        
    Returns:
        dict: Anomaly details and flags.
    """
    try:
        data = df[target_col].values
        
        # 1. Z-Score Method
        z_scores = np.abs(stats.zscore(data))
        z_anomaly = bool(z_scores[-1] > 3.0) # Threshold for anomaly
        
        # 2. Isolation Forest Method
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        # Reshape for sklearn
        data_reshaped = data.reshape(-1, 1)
        iso_forest.fit(data_reshaped)
        
        # -1 for anomaly, 1 for normal
        iso_pred = iso_forest.predict(data_reshaped)
        iso_anomaly = bool(iso_pred[-1] == -1)
        
        # Determine Severity
        severity = "Normal"
        if z_anomaly and iso_anomaly:
            severity = "Critical"
        elif z_anomaly or iso_anomaly:
            severity = "Warning"
            
        return {
            "z_score_flag": z_anomaly,
            "z_score_val": float(z_scores[-1]),
            "iso_forest_flag": iso_anomaly,
            "severity": severity,
            "message": f"Recent data indicates a {severity.lower()} surge anomaly." if severity != "Normal" else "Patient inflows are within expected statistical limits."
        }
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        return {"severity": "Unknown", "message": "Anomaly detection failed."}
