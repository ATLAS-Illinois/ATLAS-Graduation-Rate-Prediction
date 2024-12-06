# Predicting Student Graduation Rates Using Machine Learning

## Overview
This project focuses on developing a machine learning pipeline to predict the likelihood of student graduation at the University of Illinois Urbana-Champaign (UIUC). The model uses pre-enrollment features such as high school academic performance, demographic data, socioeconomic status, and parental education levels to predict graduation probabilities. The deliverable includes a Streamlit-based interface where users can input predictor values and obtain graduation predictions alongside confidence intervals.

---

## Files in the Repository

### `app3.py`
- A Python script implementing the Streamlit interface for predicting student graduation rates.
- **Features**:
  - User-friendly form input for predictors such as:
    - First-generation status.
    - Underrepresented minority (URM) status.
    - Gender.
    - High school GPA.
    - High school language years.
    - Socioeconomic indicators (e.g., ZIP income level).
  - Predicts the probability of graduation using the trained model (`best_model_final.joblib`).
  - Displays confidence intervals for predictions and visualizes these intervals.
- **Note**: Confidence intervals are calculated via bootstrapping with slight perturbations to the input data.

### `best_model_final.joblib`
- A serialized version of the final XGBoost model trained on the cleaned dataset.
- **Details**:
  - Optimized for AUC scores.
  - Features were carefully selected to balance performance and simplicity.
  - Achieved an AUC score of **0.823** on the cleaned test data.

### `ATLAS_LOTE_Project_Final.ipynb` (exported as a PDF)
- A comprehensive Jupyter Notebook detailing the entire machine learning pipeline.
- **Key Steps**:
  - **Data Cleaning**: Handled missing values and derived new predictors. Removed outliers using domain-specific rules.
  - **Feature Engineering**: Encoded categorical variables, handled imbalances, and included socioeconomic indicators from ZIP code data.
  - **Exploratory Data Analysis (EDA)**: Visualized trends in graduation rates across predictors like GPA, race, language years, and income levels.
  - **Modeling**: Experimented with various algorithms (e.g., logistic regression, random forest, XGBoost) using class-balancing techniques such as SMOTE.
  - **Model Evaluation**: Compared performance using metrics such as AUC and confusion matrices.
  - **Noise Identification**: Assessed the impact of noise by comparing AUC scores of models trained on the original vs. cleaned datasets.
  - **Feature Selection**: Selected predictors based on feature importance and logical reasoning to improve interpretability.

---

## Key Results

### 1. **Impact of Noise Removal**
- AUC score improved from **0.665** (original noisy data) to **0.823** (cleaned data).
- Removed **4,339 noisy observations** based on thresholds for GPA, language years, and other metrics.
- Demonstrated that the original dataset contained significant noise (~49% randomness).

### 2. **Model Performance**
- Cleaned data resulted in a more resilient model with higher AUC scores under various simulations.
- Final predictors were carefully selected to maximize interpretability and user-friendliness for the Streamlit app.

### 3. **Final Tool**
- The Streamlit app allows users to input hypothetical scenarios and receive predictions in a user-friendly interface.
- Confidence intervals ensure transparency about prediction reliability.

---

## How to Run the Project

### Setup
1. Install required libraries:
   - `streamlit`, `xgboost`, `joblib`, `numpy`, `pandas`, `matplotlib`, etc.
2. Place `app3.py` and `best_model_final.joblib` in the same directory.

### Launch Streamlit
1. Run the command:
   ```bash
   streamlit run app3.py



