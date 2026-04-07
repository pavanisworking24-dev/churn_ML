# Churn Risk Dashboard

Streamlit app for telecom churn prediction with:
- Interactive dashboard
- Single-customer prediction
- Batch CSV scoring

## Project Files

- `app.py`
- `requirements.txt`
- `runtime.txt`
- `best_churn_model.pkl`
- `encoded_columns.pkl`
- `optimal_threshold.pkl`
- `scaler.pkl` or `scaler (2).pkl`
- `selected_features.pkl` or `selected_features (1).pkl`

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment Notes

- Python version is set via `runtime.txt`.
- `scikit-learn` is pinned in `requirements.txt` to match model artifacts.
- Keep all model artifact files in the same folder as `app.py`.

## Expected CSV Columns

Required for dashboard and batch scoring:

`gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges`

Optional:

`customerID`, `Churn`
