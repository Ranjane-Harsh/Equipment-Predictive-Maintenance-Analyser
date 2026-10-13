# Industrial Equipment Predictive Maintenance Analyzer

A Machine Learning system built to analyze multi-sensor IoT telemetry data, detect early equipment degradation patterns, predict machine failures up to 24 hours in advance, and generate automated diagnostic maintenance alerts.

---

## 📌 Project Overview

In industrial manufacturing, unexpected machine breakdowns cause expensive factory downtime, safety hazards, and costly emergency repairs. However, machines rarely fail without warning—they emit subtle signals such as vibrating slightly more than usual, running hot, or logging non-fatal error codes before breaking down completely.

This project implements an end-to-end **Predictive Maintenance Pipeline** on real-world industrial IoT telemetry:
1. **Monitors multi-sensor readings** (voltage, rotation speed, pressure, and vibration) over time.
2. **Tracks warning logs and service history** (non-fatal error counts and days since last component replacement).
3. **Predicts impending machine breakdowns** within a 24-hour window using an optimized **XGBoost** classification model.
4. **Calibrates alerts dynamically** to avoid false alarms (preventing technician "alert fatigue").
5. **Generates actionable maintenance reports** detailing which machine is at risk, why it is flagged, and which components need inspection.

---

## ⚙️ How It Works (Pipeline Workflow)

```text
[Raw IoT Data]
 ├── Sensor Telemetry (Hourly volt, rotate, pressure, vibration)
 ├── Error Logs (Non-fatal warnings error1 - error5)
 ├── Maintenance History (Component replacements comp1 - comp4)
 └── Machine Metadata (Model and age)
        │
        ▼
[Feature Engineering]
 ├── Short-term trends (3-hour rolling averages)
 ├── Long-term trends (24-hour rolling averages)
 ├── Sensor volatility (24-hour rolling standard deviations)
 ├── Error frequency (24-hour error count windows)
 └── Component wear (Days elapsed since last part replacement)
        │
        ▼
[Machine Learning Engine]
 ├── 80/20 Chronological train-test split (Prevents future data leakage)
 ├── Square-root balanced class weighting (Handles ~1,250 : 1 extreme class imbalance)
 └── XGBoost Classifier training
        │
        ▼
[Threshold Calibration & Evaluation]
 ├── Computes Precision-Recall Curve (PR-AUC)
 └── Finds optimal decision threshold (Maximizing F1-score)
        │
        ▼
[Automated Maintenance Alerting]
 └── Flags machines crossing risk threshold & prints diagnostic summaries
```

---

## 📊 Key Results & Metrics

Predicting rare industrial failures involves extreme class imbalance (less than **0.1%** of data points are failure events). Standard metrics like accuracy or ROC-AUC can be deceptive in this setting. Our model is optimized for **Precision-Recall (PR-AUC)**:

| Metric | Score | What It Means |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.9997** | Strong ranking ability across all operating hours |
| **PR-AUC** | **0.7888** | Reliable performance on the rare failure class (vs. random baseline of 0.0008) |
| **Recall (Class 1)** | **74%** | Successfully catches nearly 3 out of every 4 impending breakdowns |
| **Precision (Class 1)** | **69%** | Alerts are reliable—roughly 7 out of 10 flagged alerts represent actual degradation |
| **F1-Score** | **0.71** | Balanced trade-off between early detection and minimizing false alarms |

---

## 🛠️ Skills & Technologies Used

- **Programming Language:** Python 3.12
- **Data Engineering & Manipulation:** Pandas, NumPy
  - Time-series rolling windows (3-hour and 24-hour rolling means and standard deviations)
  - Event-based window aggregation (24-hour rolling error counts)
  - Time-since-event tracking via fast chronological as-of joins (`pd.merge_asof`)
- **Machine Learning & Modeling:** Scikit-Learn, XGBoost
  - Gradient Boosted Decision Trees (XGBoost Classifier)
  - Extreme class imbalance handling (calibrated `scale_pos_weight`)
  - Precision-Recall curve analysis and dynamic threshold tuning
  - Model serialization and persistence (`joblib`)
- **Data Visualization & Reporting:** Matplotlib, ReportLab (automated PDF report generation)
- **Version Control & Development:** Git, GitHub, VS Code

---

## 📁 Project Structure

```text
Industrial Equipment Predictive Maintainance Analyser/
│
├── data/                                 # Raw dataset files
│   ├── PdM_telemetry.csv                 # Hourly sensor records
│   ├── PdM_errors.csv                    # Non-fatal error warning logs
│   ├── PdM_maint.csv                     # Historical component replacements
│   ├── PdM_failures.csv                  # Ground-truth machine breakdown timestamps
│   └── PdM_machines.csv                  # Machine metadata (model, age)
│
├── src/                                  # Core pipeline modules
│   ├── data_processing.py                # Data loading, cleaning & multi-scale feature engineering
│   ├── model.py                          # Chronological splitting, XGBoost training & threshold tuning
│   └── reporting.py                      # Diagnostic reporting module with root-cause indicators
│
├── main.py                               # Main pipeline entry point
├── requirements.txt                      # Python dependencies
├── generate_interview_report.py          # Script to generate technical interview study PDF
├── Interview_Report_Predictive_Maintenance.pdf # Comprehensive technical case study report
└── README.md                             # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/industrial-predictive-maintenance.git
cd industrial-predictive-maintenance
```

### 2. Set Up a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Place the Microsoft Azure Predictive Maintenance dataset files into the `data/` folder:
- `PdM_telemetry.csv`
- `PdM_errors.csv`
- `PdM_maint.csv`
- `PdM_failures.csv`
- `PdM_machines.csv`

### 5. Run the End-to-End Pipeline
```bash
python main.py
```

The script will:
1. Ingest and merge all time-series telemetry, error logs, and maintenance records.
2. Engineer 26 predictive features.
3. Train the XGBoost model on the first 80% of data (chronological split).
4. Evaluate performance on the final 20% holdout set and calculate the optimal decision threshold.
5. Save the trained model artifact to `xgboost_pdm.joblib`.
6. Run the diagnostic reporting module on recent machine telemetry and display maintenance alerts.

---

## 📋 Sample Diagnostic Alert Output

When a machine exceeds the calibrated risk threshold, the reporting module produces an alert with operational context:

```text
⚠️ URGENT MAINTENANCE ALERTS DETECTED ⚠️
============================================================
Machine ID      : 24
Timestamp       : 2015-12-31 06:00:00
Failure Risk    : 92.4%
Recent Warnings : error2 (3x), error3 (1x)
Component Age   : Comp1: 15d, Comp2: 90d, Comp3: 120d, Comp4: 30d
------------------------------------------------------------
```

This output tells technicians **which** machine needs service, **why** (elevated error frequency and component age), and **when**—allowing proactive part replacements before an unplanned shutdown occurs.

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
