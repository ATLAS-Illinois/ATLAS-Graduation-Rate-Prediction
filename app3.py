import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm
from xgboost import XGBClassifier

model = joblib.load('best_model_final.joblib')

st.title("Degree Achievement Prediction")

applied_first_gen = st.radio("Is the student a First Generation Applicant?", options=["Yes", "No"])
applied_first_gen = 1 if applied_first_gen == "Yes" else 0

applied_urm = st.radio("Is the student an Underrepresented Minority (URM)?", options=["Yes", "No"])
applied_urm = 1 if applied_urm == "Yes" else 0

gender = st.selectbox(
    "What is the gender of the student?",
    options=["Female", "Male", "Neutral"]
)
gender_f = 1 if gender == "Female" else 0

hs_lang_years = st.slider("How many years of high school language has the student completed?", min_value=0, max_value=10, value=4)

hs_gpa = st.slider("High School GPA", min_value=0.0, max_value=4.0, step=0.1)

income_level_mapping = {
    'Low': 0,
    'Lower-Middle': 1,
    'Upper-Middle': 2,
    'High': 3
}
hs_zip_income_label = st.selectbox(
    "What is the income level of the student’s high school ZIP code?", 
    options=list(income_level_mapping.keys())
)
hs_zip_income = income_level_mapping[hs_zip_income_label]

input_data = pd.DataFrame([{
    'APPLIED FIRST GENERATION': applied_first_gen,
    'APPLIED URM': applied_urm,
    'Gender_F': gender_f,
    'TOTAL HS LANG YEARS TAKEN': hs_lang_years,
    'HS GPA': hs_gpa,
    'HS ZIP Income Level Encoded': hs_zip_income,
}])

# def calculate_confidence_intervals(pred_proba, alpha=0.05):
#     n_iterations = 100
#     bootstrapped_probs = []
#     for _ in range(n_iterations):
#         perturbed_data = input_data + np.random.normal(0, 0.05, input_data.shape)
#         perturbed_proba = model.predict_proba(perturbed_data)[:, 1][0]
#         bootstrapped_probs.append(perturbed_proba)
#     mean_proba = np.mean(bootstrapped_probs)
#     std_dev = np.std(bootstrapped_probs)
    
#     confidence_intervals = {
#         '80%': (mean_proba - norm.ppf(1 - 0.2 / 2) * std_dev, mean_proba + norm.ppf(1 - 0.2 / 2) * std_dev),
#         '95%': (mean_proba - norm.ppf(1 - 0.05 / 2) * std_dev, mean_proba + norm.ppf(1 - 0.05 / 2) * std_dev),
#         '99%': (mean_proba - norm.ppf(1 - 0.01 / 2) * std_dev, mean_proba + norm.ppf(1 - 0.01 / 2) * std_dev)
#     }
#     return mean_proba, confidence_intervals


def calculate_confidence_intervals(pred_proba, alpha=0.05):
    n_iterations = 100
    bootstrapped_probs = []
    for _ in range(n_iterations):
        perturbed_data = input_data + np.random.normal(0, 0.05, input_data.shape)
        perturbed_proba = model.predict_proba(perturbed_data)[:, 1][0]
        bootstrapped_probs.append(perturbed_proba)
    
    mean_proba = np.mean(bootstrapped_probs)
    std_dev = np.std(bootstrapped_probs)
    
    confidence_intervals = {
        '80%': (
            max(0, mean_proba - norm.ppf(1 - 0.2 / 2) * std_dev),  # Clip lower bound
            min(1, mean_proba + norm.ppf(1 - 0.2 / 2) * std_dev)   # Clip upper bound
        ),
        '95%': (
            max(0, mean_proba - norm.ppf(1 - 0.05 / 2) * std_dev),
            min(1, mean_proba + norm.ppf(1 - 0.05 / 2) * std_dev)
        ),
        '99%': (
            max(0, mean_proba - norm.ppf(1 - 0.01 / 2) * std_dev),
            min(1, mean_proba + norm.ppf(1 - 0.01 / 2) * std_dev)
        )
    }
    return mean_proba, confidence_intervals

if st.button("Predict Graduation Probability"):
    prediction_proba = model.predict_proba(input_data)[:, 1][0]
    prediction = int(prediction_proba >= 0.5)
    mean_proba, confidence_intervals = calculate_confidence_intervals(prediction_proba)

    st.subheader("Prediction Results")
    if prediction == 1:
        st.success("Prediction: The student is likely to graduate 🎓")
    else:
        st.warning("Prediction: The student is less likely to graduate")
    
    st.write(f"Estimated Probability of Graduation: {mean_proba:.2f}")
    
    ci_lower_95, ci_upper_95 = confidence_intervals['95%']
    st.write(f"95% Confidence Interval: [{ci_lower_95:.2f}, {ci_upper_95:.2f}]")

    st.subheader("Confidence Interval Visualization")

    plt.figure(figsize=(8, 2))

    colors = {'80%': 'darkblue', '95%': 'cornflowerblue', '99%': 'lightblue'}
    for level, (ci_lower, ci_upper) in confidence_intervals.items():
        plt.plot([ci_lower, ci_upper], [1, 1], 
                 color=colors[level], linewidth=10 if level == '80%' else 6 if level == '95%' else 2,
                 label=f'{level} CI')
        
    plt.scatter([mean_proba], [1], color='orange', zorder=5, label=f'Mean Probability: {mean_proba:.2f}')
    plt.xlabel("Probability")
    plt.yticks([])
    plt.title("Estimated Probability with Confidence Intervals")
    plt.legend(loc='upper right', fontsize = 6)
    st.pyplot(plt)
