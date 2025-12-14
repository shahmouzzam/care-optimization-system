# AI-Driven Care Resource Optimization System

An intelligent system combining Machine Learning and Mathematical Optimization for care facility resource allocation and staff scheduling.

![Results Dashboard](results/summary_dashboard.png)

## Overview

This system optimizes staff scheduling and resource allocation for care facilities using a hybrid approach that combines predictive machine learning models with mixed-integer linear programming (MILP) optimization.

**Key Results:**
- 8% operational cost reduction
- 20% improvement in staff utilization
- 72% patient coverage with optimal resource allocation

## Features

- **ML-based Duration Prediction**: Random Forest model predicts visit durations with 5.4-minute MAE
- **Demand Forecasting**: Gradient Boosting for capacity planning and proactive scheduling
- **Multi-Constraint Optimization**: MILP solver handles skills, time windows, capacity, and cost constraints
- **Intelligent Staff Matching**: Scoring algorithm matches staff to patients based on qualifications, location, and cost
- **Comprehensive Visualization**: Automated dashboards and performance reports

## Architecture

```
Data Layer → ML Prediction → Optimization Engine → Output & Visualization
    ↓            ↓                  ↓                      ↓
Patients    Duration          Optimal Schedule        Dashboards
Staff       Demand Forecast   Cost Minimization       Metrics
Visits      Staff Matching    Constraint Solving      Reports
```

## Technical Stack

- **Python 3.8+**
- **Machine Learning**: scikit-learn (Random Forest, Gradient Boosting)
- **Optimization**: PuLP (CBC solver for MILP)
- **Data Processing**: pandas, NumPy
- **Visualization**: matplotlib, seaborn

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/care-optimization-system.git
cd care-optimization-system

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the complete optimization pipeline
python src/main.py
```

This will:
1. Generate synthetic care facility data (50 patients, 15 staff, 380 visits)
2. Train ML models for duration prediction and demand forecasting
3. Run optimization algorithms (greedy baseline + MILP)
4. Generate visualizations and performance reports
5. Save results to `results/` directory

**Runtime:** ~1 minute

## Project Structure

```
care-optimization-system/
├── src/
│   ├── data_generator.py       # Synthetic data generation
│   ├── ml_models.py            # ML prediction models
│   ├── optimization_engine.py  # MILP optimization solver
│   ├── visualizations.py       # Results visualization
│   └── main.py                 # Pipeline orchestration
├── results/
│   ├── summary_dashboard.png   # Performance dashboard
│   ├── optimized_schedule.csv  # Final schedule
│   └── [other outputs]
├── requirements.txt
└── README.md
```

## Machine Learning Components

### Duration Predictor
- **Algorithm**: Random Forest Regressor
- **Features**: Patient age, acuity, visit type, mobility, time windows, temporal features
- **Performance**: 
  - Test MAE: 5.4 minutes
  - R² Score: 0.82
  - Cross-validation MAE: 5.5 ± 0.6 minutes

### Demand Forecaster
- **Algorithm**: Gradient Boosting Regressor
- **Purpose**: Predict hourly/daily care demand for capacity planning
- **Applications**: Proactive staffing, resource allocation

### Staff-Patient Matcher
- **Algorithm**: Multi-factor scoring system
- **Factors**: Skill compatibility (30%), geographic proximity (20%), experience (10%), performance (10%), cost efficiency (penalty)

## Optimization Engine

### Problem Formulation

**Objective Function:**
```
Minimize: Total Cost = Σ(staff_rate × visit_duration) + travel_penalty
```

**Constraints:**
1. Each visit assigned exactly once
2. Staff skill level ≥ required skill level
3. Visits within patient time windows
4. No overlapping visits for same staff
5. Staff working hours ≤ contracted hours
6. Travel time between consecutive visits

**Solver:** PuLP with CBC backend (Mixed-Integer Linear Programming)

## Results

### Performance Metrics

| Metric | Baseline (Greedy) | Optimized (MILP) | Improvement |
|--------|------------------|------------------|-------------|
| Total Cost | £612/day | £563/day | 8.0% ↓ |
| Staff Utilization | 35% | 42% | 20% ↑ |
| Avg Cost/Visit | £12.24 | £11.26 | 8.0% ↓ |
| Patient Coverage | 72% | 72% | Maintained |

**Annual Savings Potential:** £17,500 for a 50-patient facility

### Output Files

Running the system generates:
- `optimized_schedule.csv` - Final staff-patient assignments
- `summary_dashboard.png` - Comprehensive results visualization
- `staff_utilization.png` - Staff usage analysis
- `cost_breakdown.png` - Cost analysis by type, staff, and time
- `patient_coverage.png` - Coverage analysis by acuity
- `schedule_timeline.png` - Gantt-chart style schedule view
- `results_summary.json` - Metrics in JSON format
- `comparison_report.csv` - Baseline vs optimized comparison

## Customization

### Adjust Problem Size

Edit `src/main.py`:
```python
results = pipeline.run_complete_pipeline(
    n_patients=50,      # Number of patients
    n_staff=15,         # Number of staff members
    optimize_days=7     # Planning horizon (days)
)
```

### Modify ML Models

Edit `src/ml_models.py`:
- Adjust Random Forest hyperparameters
- Add custom features
- Try different algorithms (XGBoost, Neural Networks)

### Customize Optimization

Edit `src/optimization_engine.py`:
- Modify objective function weights
- Add new constraints (e.g., continuity of care)
- Adjust solver parameters (time limits, gap tolerance)

## Use Cases

- **Care Facilities**: Home care, nursing homes, assisted living
- **Healthcare Providers**: Staff scheduling, capacity planning
- **Research**: Healthcare operations, optimization algorithms
- **Education**: ML + optimization case studies

## Key Algorithms

### Data Generation
Realistic synthetic data generator modeling:
- Patient demographics and acuity levels
- Staff qualifications and availability
- Visit requirements with time windows
- UK care industry regulations

### ML Pipeline
1. Feature engineering (demographic, temporal, categorical)
2. Model training with cross-validation
3. Prediction with uncertainty estimation
4. Demand forecasting for proactive planning

### Optimization Pipeline
1. Problem formulation (MILP)
2. Constraint generation
3. Solver execution (CBC)
4. Solution extraction and validation
5. Greedy baseline for comparison

## Performance Characteristics

- **Scalability**: Handles 100+ patients, 50+ staff
- **Speed**: ~1 minute for 50 patients, 15 staff, 7 days
- **Accuracy**: Duration prediction within ±10%
- **Optimality**: Within 5% of optimal solution

## Future Enhancements

- Real-time dynamic scheduling with live updates
- Deep learning for demand forecasting
- Reinforcement learning for adaptive scheduling
- Vehicle routing optimization (full TSP)
- Mobile app integration
- Multi-day rolling horizon optimization
- Staff preference learning

## Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
PuLP>=2.7.0
```

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Built with Python, scikit-learn, and PuLP**
