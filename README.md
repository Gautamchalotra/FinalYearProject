#  MediOptima – AI-Powered Hospital Resource Optimization System

<p align="center">
  <b>A Production-Ready AI Pipeline for Predictive Healthcare Operations</b><br>
  <i>Perfect for Final Year Projects, Portfolio Demonstrations, and Operational Logistics Researches.</i>
</p>

##  Project Overview
MediOptima bridges the gap between historical hospital data and real-world operational readiness. By automating time-series machine learning, generating rule-based anomaly intelligence, and computing constraints via linear programming, MediOptima ensures hospitals never suffer from unexpected bed deficits or staff shortages. 

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web_App-red.svg)
![Machine Learning](https://img.shields.io/badge/AI-Forecasting-green.svg)

##  Architecture Design

```mermaid
graph TD;
    A[Raw Historical Data] -->|Pre-Processing| B[Data Loader Utility];
    B --> C{AI Controller};
    C -->|Comparisons| D[ARIMA];
    C -->|Comparisons| E[Prophet];
    C -->|Comparisons| F[LSTM TensorFlow];
    D --> G[(Auto Best Selection)];
    E --> G;
    F --> G;
    G --> H[PuLP Optimizer];
    H --> |Minimizes Cost / Staff Ratio| I[Prescriptive Allocations];
    G --> J[NLP Insight Generator];
    B --> K[Z-Score & Isolation Forest];
    K --> L[Surge Anomaly Flags];
    I --> M[Flask Web App & Optional REST API];
    J --> M;
    L --> M;
```

##  Analytical Metrics Tracked 
All AI pipelines are continuously benchmarked avoiding model decay.
- **MAPE** (Mean Absolute Percentage Error) is the primary ranking factor determining which ML code powers the dashboard dynamically.
- **RMSE & MAE** supplied for total transparency.

##  Setup & Execution

### 1. Requirements & Clone
Ensure you have Python 3.9+ installed and `pip` working.

```bash
git clone https://github.com/your-username/MediOptima.git
cd MediOptima
```

### 2. Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows users: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prime the AI (Generate Data)
MediOptima requires a working dataset to analyze. Run the internal script to simulate exactly 365 days of a hospital ecosystem (Seasonal flows, outbreaks, holidays).
```bash
python scripts/generate_data.py
```

### 4. Deploy the Dashboard
```bash
python app.py
```

##  Modularity and Customization
* Modules are purely decoupled. 
* Look into `config/config.py` to change logic like `DOC_TO_PATIENT_RATIO` directly without digging into raw scripts.

##  Planned Roadmap
- PDF Generation Reports using `ReportLab`.
- Full Dockerization `Dockerfile` setup.

---
*Created as part of an Advanced System Architecture and Machine Learning initiative.*
