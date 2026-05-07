"""
Optimization module using PuLP to calculate optimal staff schedules minimizing cost.
"""

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value
from pulp import PULP_CBC_CMD
import logging
import math
import sys
import os

# Add config dir to path for importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config

logger = logging.getLogger(__name__)

def optimize_staff(patients: int, icu_patients: int) -> dict:
    """
    Minimizes staff cost using Linear Programming based on patient constraints.
    Returns structured dict without printing noisy solver logs.
    """
    try:
        prob = LpProblem("Staffing_Optimization", LpMinimize)
        
        doctors = LpVariable("Doctors", lowBound=0, cat='Integer')
        nurses = LpVariable("Nurses", lowBound=0, cat='Integer')
        
        doc_cost = config.STAFF_COST.get("doctor", 1000)
        nurse_cost = config.STAFF_COST.get("nurse", 500)
        prob += lpSum([doc_cost * doctors, nurse_cost * nurses]), "Total_Staff_Cost"
        
        min_doctors = math.ceil(patients / config.DOC_TO_PATIENT_RATIO)
        min_nurses = math.ceil(icu_patients / config.NURSE_TO_ICU_RATIO) + math.ceil((patients - icu_patients) / 10)
        
        prob += doctors >= min_doctors, "Min_Doctors_Constraint"
        prob += nurses >= min_nurses, "Min_Nurses_Constraint"
        
        # Suppress noisy standard output from solver
        prob.solve(PULP_CBC_CMD(msg=False))
        
        status = LpStatus[prob.status]
        
        return {
            "status": status,
            "optimal_doctors": int(value(doctors)) if value(doctors) is not None else min_doctors,
            "optimal_nurses": int(value(nurses)) if value(nurses) is not None else min_nurses,
            "total_cost": value(prob.objective) if value(prob.objective) is not None else 0
        }
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        return {"status": "Error", "message": str(e)}

def calculate_bed_requirements(predicted_patients: int, expected_discharges: int) -> dict:
    """Calculates total required beds considering safety buffer."""
    try:
        buffer = config.EMERGENCY_BUFFER
        required_beds = predicted_patients - expected_discharges
        if required_beds < 0: required_beds = 0
            
        total_required_with_buffer = math.ceil(required_beds * (1 + buffer))
        return {
            "predicted_patients": predicted_patients,
            "expected_discharges": expected_discharges,
            "required_beds": total_required_with_buffer
        }
    except Exception as e:
        logger.error(f"Bed calculation error: {e}")
        return {"status": "Error", "message": str(e)}
