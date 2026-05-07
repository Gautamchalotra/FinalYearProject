"""
Insight generator utility.
Converts numerical predictions and metrics into human-readable text insights.
"""

import logging

logger = logging.getLogger(__name__)

def generate_forecast_insights(current_patients: int, predicted_patients: int, 
                               current_icu: int, max_icu: int,
                               required_beds: int, available_beds: int,
                               required_staff: int, available_staff: int) -> list:
    """
    Generates rule-based natural language insights based on predictions.
    Ensures that meaningful insights are always returned.
    """
    insights = []
    try:
        # 1. Patient trend insight
        diff_pct = ((predicted_patients - current_patients) / current_patients) * 100 if current_patients > 0 else 0
        if diff_pct > 10:
            insights.append(f" Patient inflow is expected to increase by {diff_pct:.1f}% in the upcoming period.")
        elif diff_pct < -10:
            insights.append(f" Patient inflow is expected to decrease by {abs(diff_pct):.1f}%.")
        else:
            insights.append(" Patient volume is expected to remain relatively stable.")
            
        # 2. ICU capacity insight
        icu_usage_pct = (current_icu / max_icu) * 100 if max_icu > 0 else 0
        if icu_usage_pct > 80:
            insights.append(f" Critical Alert: ICU capacity is dangerously high at {icu_usage_pct:.1f}%. Immediate action may be required.")
        elif icu_usage_pct > 60:
            insights.append(f" Warning: ICU capacity is at {icu_usage_pct:.1f}%. Consider adding reserve staff.")
        else:
            insights.append(" ICU capacity is currently within safe operational limits.")
            
        # 3. Bed Utilization
        if required_beds > available_beds:
            insights.append(f" Bed utilization will exceed maximum capacity by {required_beds - available_beds} beds.")
        elif required_beds > available_beds * 0.8:
            insights.append(f" Bed utilization is nearing maximum capacity (over 80%).")
            
        # 4. Staffing Requirements
        if required_staff > available_staff:
            insights.append(f" Critical Staffing Alert: Required staff ({required_staff}) exceeds currently scheduled staff ({available_staff}).")
        
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        
    # Fallback to ensure there is always an output
    if not insights:
        insights.append(" No major risks detected. Hospital operations are stable.")
        
    return insights
