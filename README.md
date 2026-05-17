# AquaMind – AI Water Optimization for Data Centers

AI-powered sustainability and cooling optimization platform for modern data centers using Machine Learning, climate-aware simulation, and intelligent workload optimization.

---

## Overview

AquaMind is a real-time AI decision-support system designed to help data centers reduce cooling-water consumption while maintaining operational efficiency.

The platform predicts cooling-water demand using machine learning models trained on realistic synthetic telemetry data and recommends optimization strategies based on:

- Climate conditions
- Cooling architecture
- Server utilization
- Regional water stress
- Workload intensity

The system combines:
- Predictive analytics
- Sustainability intelligence
- Optimization simulation
- Explainable AI dashboards

---

## Key Features

### AI-Based Water Usage Prediction

Predicts hourly cooling-water consumption using:
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost (optional)

---

### Climate-Aware Optimization

Analyzes:
- Outside temperature
- Humidity
- Server load
- Cooling infrastructure
- Regional water stress

Then recommends:
- Cooling strategy shifts
- Hybrid cooling transitions
- Workload balancing
- Process optimization

---

### Sustainability Analytics

Tracks:
- Water Usage Effectiveness (WUE)
- Annual water savings
- CO₂ reduction
- Energy savings
- Global sustainability impact

---

### Interactive Dashboard

Built using Streamlit with:
- Real-time KPI cards
- Feature importance visualization
- Optimization comparison charts
- Water vs temperature analysis
- Scenario simulation controls

---

## Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Dashboard | Streamlit |
| ML Models | scikit-learn, XGBoost |
| Data Processing | NumPy, Pandas |
| Visualization | Altair |
| Optimization Logic | Custom AI Decision Engine |
| Data Generation | Synthetic Telemetry Simulation |

---

# Architecture

```text
User Inputs
   ↓
Synthetic Telemetry Generator
   ↓
Feature Engineering
   ↓
ML Training Pipeline
   ↓
Best Model Selection
   ↓
Optimization Engine
   ↓
Sustainability Analytics
   ↓
Interactive Dashboard
```

---

# Machine Learning Pipeline

The application automatically:
1. Generates realistic synthetic data
2. Trains multiple regression models
3. Evaluates performance using:
   - R² Score
   - RMSE
   - MAE
   - Cross-validation
4. Selects the best-performing model
5. Uses the model for optimization simulation

---

## Dataset Features

| Feature | Description |
|---|---|
| outside_temperature | Ambient temperature |
| humidity | Relative humidity |
| server_load | Data center utilization |
| power_usage_kw | Power consumption |
| cooling_type | Cooling architecture |
| workload_intensity | Compute intensity |
| region | Water stress region |
| region_water_stress_index | Water scarcity factor |

### Target Variable

```text
water_usage_liters
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/Mohit-analytics/AquaMind_Project.git
cd aquamind
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas streamlit altair scikit-learn xgboost
```

---

# Run the Application

```bash
streamlit run app.py
```

The dashboard will launch automatically in your browser.

Default:
```text
http://localhost:8501
```

---

# Project Structure

```text
AquaMind/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── synthetic datasets
│
├── models/
│   └── trained model artifacts
│
├── visuals/
│   └── charts and screenshots
│
└── docs/
    └── architecture and reports
```

---

# Optimization Logic

The optimization engine performs:

## Cooling Strategy Recommendation

Switches between:
- Water-cooled
- Hybrid cooling
- Air-cooled

based on:
- Temperature
- Humidity
- Water stress

---

## Workload Optimization

Suggests:
- Cross-region load balancing
- Workload smoothing
- Off-peak scheduling
- Server consolidation

---

# Sustainability Metrics

AquaMind estimates:

- Annual water savings
- Annual CO₂ reduction
- Annual energy savings
- ESG sustainability impact
- Global scaling projections

---

# Example Outputs

## Predicted Metrics

- Baseline water usage
- Optimized water usage
- Water savings percentage
- WUE improvement

## Sustainability Impact

- CO₂ reduction
- Annual cost savings
- Energy reduction
- Global cloud infrastructure impact

---

# Visualization Components

The dashboard includes:

- Feature importance charts
- Water vs climate analysis
- Before vs after optimization comparison
- Real-time KPI cards
- Sustainability impact panels

---

# Future Improvements

- Real telemetry integration
- Kubernetes workload orchestration
- Reinforcement Learning scheduling
- GPU telemetry monitoring
- Carbon-aware optimization
- Multi-region intelligent orchestration
- Live cloud deployment

---

# Research & Engineering Concepts

This project combines concepts from:

- Machine Learning
- Green Computing
- Data Center Engineering
- Sustainable AI
- Climate Analytics
- Systems Optimization
- Resource Scheduling

---

# Why AquaMind Matters

Modern data centers consume massive amounts of:
- Water
- Energy
- Cooling resources

AquaMind demonstrates how AI can help:
- Reduce environmental impact
- Improve operational efficiency
- Support sustainability goals
- Enable climate-aware infrastructure

---



# Authors

Mohit Sharma  
AI & Sustainability Engineering Enthusiast

---

# License

MIT License

---

# Acknowledgements

Libraries and frameworks used:
- Streamlit
- scikit-learn
- XGBoost
- NumPy
- Pandas
- Altair

Research inspiration:
- Sustainable data center optimization
- AI-based resource management
- Water Usage Effectiveness (WUE)
- Green cloud computing systems
