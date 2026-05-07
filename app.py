"""
MediOptima - Flask Web Application
Provides interactive UI to visualize forecasts, optimize staff, and detect anomalies.
Run with: python app.py
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import pandas as pd
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

from config import config
from utils.data_loader import load_data, get_recent_metrics
from services.forecasting import get_all_forecasts
from services.optimization import optimize_staff, calculate_bed_requirements
from services.anomaly import detect_anomalies
from services.medicine import predict_medicine_demand, calculate_medicine_cost
from utils.insights import generate_forecast_insights

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medioptima_secret_key'

# Store logs in memory for the session
exec_logs = []

def log_msg(msg):
    exec_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

@app.context_processor
def inject_hospital():
    return dict(current_hospital=session.get('hospital_id', 'HOSP-A'), available_hospitals=['HOSP-A', 'HOSP-B'])

@app.route('/set_hospital', methods=['POST'])
def set_hospital():
    session['hospital_id'] = request.form.get('hospital_id', 'HOSP-A')
    return redirect(request.referrer or url_for('home'))

def get_hospital_data():
    df = load_data(config.DATA_PATH)
    hospital_id = session.get('hospital_id', 'HOSP-A')
    df = df[df['hospital_id'] == hospital_id].reset_index(drop=True)
    return df

@app.route('/')
def home():
    df = get_hospital_data()
    if df.empty:
        return "Missing Data Error: Run `python scripts/generate_data.py` before starting.", 500
    
    latest = get_recent_metrics(df)
    log_msg("System Initialized.")
    log_msg(f"Data Loaded: {len(df)} records retrieved.")

    return render_template('home.html', latest=latest, logs=exec_logs)

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    df = get_hospital_data()
    forecast_days = config.FORECAST_DAYS

    # Handle form submission for forecast days
    if request.method == 'POST':
        try:
            forecast_days = int(request.form.get('forecast_days', config.FORECAST_DAYS))
        except ValueError:
            forecast_days = config.FORECAST_DAYS

    all_models = get_all_forecasts(df, forecast_days=forecast_days)
    best_model = next(m for m in all_models if m['is_best'])
    predictions = best_model['predictions']
    log_msg(f"Models trained successfully. Winning model: {best_model['name']}.")

    # Smooth Predictions
    import numpy as np
    smoothed_predictions = pd.Series(predictions).rolling(window=3, min_periods=1).mean().values
    
    # Generate Plotly graph
    hist_plotted = df.tail(60) # Keep visual clean
    last_hist_val = hist_plotted['patients'].iloc[-1]
    last_hist_date = hist_plotted['date'].iloc[-1]
    
    # Limit unrealistic jumps
    clipped_predictions = np.clip(smoothed_predictions, last_hist_val * 0.5, last_hist_val * 1.5)
    
    future_dates = [last_hist_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    combined_dates = list(hist_plotted['date']) + future_dates
    combined_values = list(hist_plotted['patients']) + list(clipped_predictions)
    
    fig = go.Figure()
    
    # One continuous line
    fig.add_trace(go.Scatter(
        x=combined_dates, 
        y=combined_values, 
        name='Patient Trend', 
        mode='lines+markers', 
        line=dict(color='#3b82f6', width=3)
    ))
    
    # Highlight forecast area using shapes
    fig.add_vrect(
        x0=last_hist_date, x1=future_dates[-1],
        fillcolor="rgba(239, 68, 68, 0.1)", opacity=0.5,
        layer="below", line_width=0,
        annotation_text="Forecast Region", annotation_position="top left"
    )
    
    fig.update_layout(
        height=400, 
        hovermode="x unified", 
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=20)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(title_text="Patient Volume", showgrid=True, gridcolor='rgba(128,128,128,0.2)')
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    metrics_data = []
    for m in all_models:
        metrics_data.append({
            "Model": m['name'] + ("  (Selected)" if m['is_best'] else ""),
            "MAPE": f"{m['metrics']['MAPE']*100:.2f}%",
            "RMSE": f"{m['metrics']['RMSE']:.2f}",
            "MAE": f"{m['metrics']['MAE']:.2f}"
        })

    return render_template('forecasting.html', 
        graphJSON=graphJSON, 
        models=metrics_data, 
        forecast_days=forecast_days, 
        best_model_name=best_model['name']
    )

@app.route('/optimize', methods=['GET', 'POST'])
def optimize():
    df = get_hospital_data()
    latest = get_recent_metrics(df)
    
    all_models = get_all_forecasts(df, forecast_days=config.FORECAST_DAYS)
    best_model = next(m for m in all_models if m['is_best'])
    predictions = best_model['predictions']
    
    sim_multiplier = 0
    is_surge = False
    
    if request.method == 'POST':
        try:
            sim_multiplier = int(request.form.get('sim_multiplier', 0))
        except ValueError:
            sim_multiplier = 0
        is_surge = 'is_surge' in request.form
        
    next_patients_base = int(predictions[0])
    next_patients_sim = int(next_patients_base * (1 + (sim_multiplier / 100.0)))
    
    if is_surge:
        est_icu_sim = int(next_patients_sim * 0.30)        # 30% to ICU under mass casualty
        est_discharges_sim = int(next_patients_sim * 0.50) # Discharges paralyze during crisis
    else:
        est_icu_sim = int(next_patients_sim * 0.15)
        est_discharges_sim = int(next_patients_sim * 0.85)

    base_opt = optimize_staff(next_patients_base, int(next_patients_base * 0.15))
    sim_opt = optimize_staff(next_patients_sim, est_icu_sim)
    bed_opt = calculate_bed_requirements(next_patients_sim, est_discharges_sim)
    
    # Calculate simulated medicine needs and costs
    sim_medicine_needs = predict_medicine_demand(next_patients_sim, est_icu_sim, is_surge)
    medicine_cost_data = calculate_medicine_cost(sim_medicine_needs)
    
    staff_cost = sim_opt.get('total_cost', 0)
    medicine_cost = medicine_cost_data['total_medicine_cost']
    total_operational_cost = staff_cost + medicine_cost
    
    # Emergency Response Planner - Detect Deficits
    emergency_deficits = {
        'is_emergency': is_surge or (bed_opt['required_beds'] > latest['available_beds']),
        'extra_beds': max(0, bed_opt['required_beds'] - latest['available_beds']),
        'extra_doctors': max(0, sim_opt['optimal_doctors'] - latest['doctors']),
        'extra_nurses': max(0, sim_opt['optimal_nurses'] - latest['nurses'])
    }
    
    log_msg("LP Optimization solver successfully cycled for both Normal and Simulated constraints.")

    # Charts for UI
    fig_icu = go.Figure()
    fig_icu.add_trace(go.Bar(name='Expected ICU Need', x=['Demand vs Capacity'], y=[est_icu_sim], marker_color='orange'))
    fig_icu.add_trace(go.Bar(name='Physical Limit', x=['Demand vs Capacity'], y=[latest['available_icu_beds']], marker_color='lightgray'))
    fig_icu.update_layout(barmode='group', title="ICU Stress Level", height=300)
    graphJSON_icu = json.dumps(fig_icu, cls=plotly.utils.PlotlyJSONEncoder)

    fig_beds = go.Figure()
    fig_beds.add_trace(go.Bar(name='Required Beds', x=['Beds Needed'], y=[bed_opt['required_beds']], marker_color='#17a2b8'))
    fig_beds.add_trace(go.Bar(name='Max Capacity', x=['Beds Needed'], y=[latest['available_beds']], marker_color='#343a40'))
    fig_beds.update_layout(barmode='group', title="Physical Bed Limits", height=300)
    graphJSON_beds = json.dumps(fig_beds, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('optimization.html', 
        next_patients_sim=next_patients_sim,
        sim_multiplier=sim_multiplier,
        is_surge=is_surge,
        bed_opt=bed_opt,
        sim_opt=sim_opt,
        base_opt=base_opt,
        latest=latest,
        graphJSON_icu=graphJSON_icu,
        graphJSON_beds=graphJSON_beds,
        sim_medicine_needs=sim_medicine_needs,
        staff_cost=staff_cost,
        medicine_cost=medicine_cost,
        total_operational_cost=total_operational_cost,
        emergency_deficits=emergency_deficits
    )

@app.route('/anomaly')
def anomaly():
    df = get_hospital_data()
    anomaly_result = detect_anomalies(df)
    return render_template('anomaly.html', anomaly=anomaly_result)

@app.route('/insights')
def insights():
    df = get_hospital_data()
    latest = get_recent_metrics(df)
    all_models = get_all_forecasts(df, forecast_days=config.FORECAST_DAYS)
    best_model = next(m for m in all_models if m['is_best'])
    predictions = best_model['predictions']
    
    next_patients = int(predictions[0])
    staff_opt = optimize_staff(next_patients, int(next_patients * 0.15))
    bed_opt = calculate_bed_requirements(next_patients, int(next_patients * 0.85))
    
    required_staff = staff_opt.get('optimal_doctors', 0) + staff_opt.get('optimal_nurses', 0)
    available_staff = latest['doctors'] + latest['nurses']

    mean_pred_patients = int(sum(predictions) / len(predictions))
    insight_messages = generate_forecast_insights(
        current_patients=latest['patients'], 
        predicted_patients=mean_pred_patients, 
        current_icu=latest['icu_patients'], 
        max_icu=latest['available_icu_beds'],
        required_beds=bed_opt['required_beds'], 
        available_beds=latest['available_beds'],
        required_staff=required_staff, 
        available_staff=available_staff
    )

    return render_template('insights.html', insights=insight_messages)

@app.route('/medicine')
def medicine():
    df = get_hospital_data()
    latest = get_recent_metrics(df)
    
    all_models = get_all_forecasts(df, forecast_days=config.FORECAST_DAYS)
    best_model = next(m for m in all_models if m['is_best'])
    predictions = best_model['predictions']
    
    predicted_patients = int(predictions[0])
    predicted_icu = int(predicted_patients * 0.15)
    
    # Simple check for surge based on 20% increase
    is_surge = predicted_patients > (latest['patients'] * 1.2)
    
    medicine_needs = predict_medicine_demand(predicted_patients, predicted_icu, is_surge)
    
    # Generate Plotly Chart
    fig = go.Figure(data=[
        go.Bar(name='Antibiotics', x=['Medicine'], y=[medicine_needs['antibiotics']], marker_color='#0d6efd'),
        go.Bar(name='Painkillers', x=['Medicine'], y=[medicine_needs['painkillers']], marker_color='#198754'),
        go.Bar(name='ICU Drugs', x=['Medicine'], y=[medicine_needs['icu_drugs']], marker_color='#dc3545')
    ])
    fig.update_layout(
        barmode='group', 
        title="Predicted Medicine Demand (Units)", 
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Generate Plotly Pie Chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Antibiotics', 'Painkillers', 'ICU Drugs'],
        values=[medicine_needs['antibiotics'], medicine_needs['painkillers'], medicine_needs['icu_drugs']],
        marker=dict(colors=['#3b82f6', '#10b981', '#ef4444']),
        hole=0.4
    )])
    fig_pie.update_layout(
        title="Medicine Demand Distribution (%)",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    graphJSON_pie = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Growth vs Past
    past_total = latest.get('total_medicine', medicine_needs['total'])
    if past_total > 0:
        growth_pct = ((medicine_needs['total'] - past_total) / past_total) * 100
    else:
        growth_pct = 0
        
    return render_template('medicine.html', 
        medicine=medicine_needs, 
        graphJSON=graphJSON, 
        graphJSON_pie=graphJSON_pie,
        predicted_patients=predicted_patients,
        growth_pct=growth_pct
    )

@app.errorhandler(404)
def page_not_found(e):
    return "404 Error: Route not found. Please check the URL.", 404

@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if hasattr(e, 'code'):
        return f"Error {e.code}: {str(e)}", e.code
    return f"500 Error: An unexpected error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
