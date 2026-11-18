# 🫀 Heart Disease Risk Predictor
BDA700 Health Analytics, Milestone 2 · Group 3: Vidhi Parmar, Sam Alavi

**Live app:** https://heart-disease-risk-grp-3.streamlit.app

Predicts the probability of heart disease from 13 clinical measurements using a logistic regression model
trained on the UCI Heart Disease (Cleveland) dataset (303 patients; target = `num > 0`).

## Files
- `classification_app.py`: Streamlit app
- `classification_model.joblib`: trained pipeline (preprocessing + logistic regression)
- `requirements.txt`: pinned package versions
- `heart_cleveland.csv`: dataset (used for the Milestone 1 chart)
- `Project_mil_2_Vidhi_Sam.ipynb`: data prep, model comparison, evaluation

## Model
- Compared Logistic Regression, Random Forest, SVC and KNN with GridSearchCV and stratified 5-fold CV.
  All were close; Logistic Regression was chosen for its top score and interpretability.
- Cross-validated ROC-AUC **0.91**. Screening threshold **0.33**, chosen for at least 85% recall.
- Test set: recall **96%**, precision **73%**.
- Links to Milestone 1: lower peak heart rate raises predicted risk, and the app plots each patient on our "Flat-Line Trap" chart.

## Run locally
```bash
pip install -r requirements.txt
streamlit run classification_app.py
```

*Educational project, not a medical device.*