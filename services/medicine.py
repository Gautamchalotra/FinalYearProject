"""
Medicine Demand Prediction Service
Predicts the required medicine units based on forecasted patient and ICU admissions.
"""

def predict_medicine_demand(predicted_patients, predicted_icu, is_surge=False):
    """
    Predicts medicine needs using a simple rule-based scale from historical patterns.
    
    Args:
        predicted_patients (int): Forecasted total patients.
        predicted_icu (int): Forecasted ICU patients.
        is_surge (bool): Whether a surge event (anomaly/emergency) is expected.
        
    Returns:
        dict: Breakdown of medicine requirements.
    """
    
    # Base historical multipliers (derived from generation logic means)
    base_antibiotics_per_patient = 1.5
    base_painkillers_per_patient = 0.65
    base_painkillers_per_emergency = 1.25
    base_icu_drugs_per_icu_patient = 3.0
    
    # Estimate emergencies based on total patients
    estimated_emergencies = int(predicted_patients * 0.25)
    
    # Surge multipliers
    surge_multiplier = 1.4 if is_surge else 1.0
    
    # Calculate estimations
    antibiotics_needed = int(predicted_patients * base_antibiotics_per_patient * surge_multiplier)
    painkillers_needed = int((predicted_patients * base_painkillers_per_patient) + (estimated_emergencies * base_painkillers_per_emergency * surge_multiplier))
    icu_drugs_needed = int(predicted_icu * base_icu_drugs_per_icu_patient * surge_multiplier)
    
    total_medicine = antibiotics_needed + painkillers_needed + icu_drugs_needed
    
    return {
        "antibiotics": antibiotics_needed,
        "painkillers": painkillers_needed,
        "icu_drugs": icu_drugs_needed,
        "total": total_medicine
    }

def calculate_medicine_cost(medicine_needs):
    """
    Calculates the operational cost of predicted medicine needs in INR.
    
    Args:
        medicine_needs (dict): Breakdown of medicine requirements.
        
    Returns:
        dict: Breakdown of costs and total cost.
    """
    # INR values per unit
    cost_antibiotics = 10
    cost_painkillers = 5
    cost_icu_drugs = 50
    
    antibiotics_cost = medicine_needs.get('antibiotics', 0) * cost_antibiotics
    painkillers_cost = medicine_needs.get('painkillers', 0) * cost_painkillers
    icu_drugs_cost = medicine_needs.get('icu_drugs', 0) * cost_icu_drugs
    
    total_medicine_cost = antibiotics_cost + painkillers_cost + icu_drugs_cost
    
    return {
        "antibiotics_cost": antibiotics_cost,
        "painkillers_cost": painkillers_cost,
        "icu_drugs_cost": icu_drugs_cost,
        "total_medicine_cost": total_medicine_cost
    }
