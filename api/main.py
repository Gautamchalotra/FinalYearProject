"""
Optional FastAPI backend for MediOptima.
Provides RESTful endpoints for forecasting and optimization.
To run: uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Ensure the parent directory is in path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.forecasting import get_best_forecast
from services.optimization import optimize_staff, calculate_bed_requirements
from services.anomaly import detect_anomalies
from utils.data_loader import load_data
from config import config

app = FastAPI(title="MediOptima API", description="AI-Powered Hospital Resource Optimization System")

class PredictResponse(BaseModel):
    model_name: str
    predictions: list
    metrics: dict

class OptimizeResponse(BaseModel):
    required_beds: int
    optimal_doctors: int
    optimal_nurses: int
    total_cost: float

@app.get("/")
def health_check():
    return {"status": "ok", "message": "MediOptima API is running."}

@app.get("/predict", response_model=PredictResponse)
def predict_patients():
    try:
        df = load_data(config.DATA_PATH)
        best_model = get_best_forecast(df, forecast_days=config.FORECAST_DAYS)
        return {
            "model_name": best_model["name"],
            "predictions": best_model["predictions"],
            "metrics": best_model["metrics"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/optimize", response_model=OptimizeResponse)
def optimize_resources():
    try:
        df = load_data(config.DATA_PATH)
        best_model = get_best_forecast(df, forecast_days=config.FORECAST_DAYS)
        next_day_predicted = int(best_model['predictions'][0])
        
        # Estimate ICU and discharges based on historical rates for simplicity
        icu_predicted = int(next_day_predicted * 0.15)
        discharges_predicted = int(next_day_predicted * 0.85)
        
        bed_reqs = calculate_bed_requirements(next_day_predicted, discharges_predicted)
        staff_reqs = optimize_staff(next_day_predicted, icu_predicted)
        
        return {
            "required_beds": bed_reqs.get("required_beds", 0),
            "optimal_doctors": staff_reqs.get("optimal_doctors", 0),
            "optimal_nurses": staff_reqs.get("optimal_nurses", 0),
            "total_cost": staff_reqs.get("total_cost", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
