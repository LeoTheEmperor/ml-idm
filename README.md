# NSL-KDD IDS Dashboard

A Streamlit dashboard for the Multiclass Intrusion Detection System trained on NSL-KDD.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Pages

| Page | What it shows |
|---|---|
| Overview | KPI cards (Macro F1, AUC, Accuracy) + model comparison bar |
| Model Comparison | Grouped per-class F1 bar + radar chart across all 4 models |
| Per-Class Analysis | Precision/Recall/F1 per class + P-R trade-off scatter |
| Confusion Matrix | Raw counts + normalised heatmaps side by side |
| Class Imbalance | Train/test distributions, SMOTE before/after, IR stats |
| Live Inference | Upload CSV → get predictions (auto-uses saved .pkl models) |

## Wiring your real models

After training in your notebook, save artifacts:

```python
import pickle
pickle.dump(xgb,    open('models/xgboost.pkl', 'wb'))
pickle.dump(rf,     open('models/random_forest.pkl', 'wb'))
pickle.dump(svm,    open('models/svm_(rbf).pkl', 'wb'))
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(le,     open('models/label_encoder.pkl', 'wb'))
```

Place them in the `models/` folder. The dashboard detects them automatically.

## Updating hardcoded metrics

In `app.py`, update the `RESULTS` dict with your actual classification report numbers:
- `precision`, `recall`, `f1` → list of 5 values in `[DoS, Normal, Probe, R2L, U2R]` order
- `macro_f1`, `accuracy`, `macro_auc` → scalar floats

Also replace the `CM_DATA` numpy arrays with your actual confusion matrices saved from `sklearn.metrics.confusion_matrix`.
