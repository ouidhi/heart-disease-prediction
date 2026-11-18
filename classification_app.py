
# imports
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# page setup 
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="🫀")


# load the model and the data (once, then cached) 
@st.cache_resource
def load_model():
    return joblib.load("classification_model.joblib")


@st.cache_data
def load_data():
    df = pd.read_csv("heart_cleveland.csv")
    df["target"] = (df["num"] > 0).astype(int)
    return df


model = load_model()
data = load_data()

THRESHOLD = 0.33
FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca",
            "sex", "fbs", "exang", "cp", "restecg", "slope", "thal"]

# title
st.title("🫀 Heart Disease Risk Predictor")
st.write("Predicts the probability of heart disease from 13 clinical measurements, using a logistic "
         "regression model trained on the UCI Heart Disease (Cleveland) dataset. "
         "BDA700 Group 3: Vidhi Parmar & Sam Alavi.")
st.caption("Educational project, not a medical device.")

# inputs in the sidebar (users see words, the model gets numbers)
CP = {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic (no chest pain)": 4}
RESTECG = {"Normal": 0, "ST-T wave abnormality": 1, "Left ventricular hypertrophy": 2}
SLOPE = {"Upsloping": 1, "Flat": 2, "Downsloping": 3}
THAL = {"Normal": 3, "Fixed defect": 6, "Reversible defect": 7}
YES_NO = {"No": 0, "Yes": 1}

st.sidebar.header("Patient details")
age = st.sidebar.number_input("Age (years)", min_value=18, max_value=100, value=55)
sex = st.sidebar.radio("Sex", ["Female", "Male"])
cp = st.sidebar.selectbox("Chest pain type", list(CP))
trestbps = st.sidebar.number_input("Resting blood pressure (mm Hg)", 80, 220, 130)
chol = st.sidebar.number_input("Cholesterol (mg/dl)", 100, 600, 240)
fbs = st.sidebar.radio("Fasting blood sugar above 120 mg/dl?", list(YES_NO))
restecg = st.sidebar.selectbox("Resting ECG", list(RESTECG))
thalach = st.sidebar.number_input("Maximum heart rate during exercise (bpm)", 60, 220, 150)
exang = st.sidebar.radio("Chest pain during exercise?", list(YES_NO))
oldpeak = st.sidebar.number_input("ST depression during exercise (oldpeak)", 0.0, 6.5, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope of the ST segment", list(SLOPE))
ca = st.sidebar.selectbox("Major vessels coloured by fluoroscopy", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thallium stress test", list(THAL))

predict = st.sidebar.button("Predict risk", type="primary")

# prediction 
if predict:
    patient = pd.DataFrame([{
        "age": age, "sex": 1 if sex == "Male" else 0, "cp": CP[cp], "trestbps": trestbps,
        "chol": chol, "fbs": YES_NO[fbs], "restecg": RESTECG[restecg], "thalach": thalach,
        "exang": YES_NO[exang], "oldpeak": oldpeak, "slope": SLOPE[slope], "ca": ca, "thal": THAL[thal],
    }])[FEATURES]

    prob = model.predict_proba(patient)[0, 1]

    st.subheader("Result")
    if prob >= THRESHOLD:
        st.error(f"⚠️ Heart Disease Risk: predicted probability {prob:.0%}")
    else:
        st.success(f"✅ Normal / Low Risk: predicted probability {prob:.0%}")
    st.progress(float(prob))
    st.caption(f"Patients at or above {THRESHOLD:.0%} are flagged. We use a low threshold on purpose: "
               "in screening, missing a sick patient is worse than a false alarm.")

    # milestone 1 chart with this patient on it 
    st.subheader("Our Milestone 1 finding: the Flat-Line Trap")
    st.write("Without heart disease, maximum heart rate drops steadily with age. With heart disease it starts "
             "lower and stays flat. The star shows this patient.")

    fig, ax = plt.subplots(figsize=(7, 4))
    for target, color, label in [(0, "#2a78d6", "No heart disease"), (1, "#eb6834", "Heart disease")]:
        group = data[data["target"] == target]
        ax.scatter(group["age"], group["thalach"], color=color, alpha=0.3, s=15)
        m, b = np.polyfit(group["age"], group["thalach"], 1)
        ax.plot(group["age"], m * group["age"] + b, color=color, linewidth=2.5, label=label)
    ax.scatter(age, thalach, marker="*", s=300, color="black", label="This patient", zorder=3)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Maximum heart rate (bpm)")
    ax.legend()
    st.pyplot(fig)

    healthy = data[data["target"] == 0]
    m, b = np.polyfit(healthy["age"], healthy["thalach"], 1)
    expected = m * age + b
    st.write(f"At age {age}, people without heart disease reach about **{expected:.0f} bpm** on average. "
             f"This patient reached **{thalach} bpm** ({thalach - expected:+.0f} bpm).")
else:
    st.info("👈 Enter the patient's details in the sidebar and press **Predict risk**.")

# model summary 
st.divider()
st.subheader("About the model")
col1, col2, col3 = st.columns(3)
col1.metric("ROC-AUC (cross-validated)", "0.91")
col2.metric("Recall (test set)", "96%")
col3.metric("Precision (test set)", "73%")
st.write("We compared Logistic Regression, Random Forest, SVC and KNN with 5-fold cross-validation. "
         "All scored about the same, so we chose Logistic Regression because it is the easiest to interpret. "
         "Recall is the share of sick patients the model catches; precision is the share of flagged "
         "patients who are actually sick.")
