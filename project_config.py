from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "Datasets"
MODELS_DIR = BASE_DIR / "Src"
HOME_GIF = BASE_DIR / "home.gif"


MODEL_FILES = {
    "diabetes": "diabetes_prediction_model.sav",
    "heart": "heart_prediction_model.sav",
    "parkinsons": "parkinsons_prediction_model.sav",
    "kidney": "kidney_prediction_model.sav",
}


APP_SECTIONS = [
    {
        "menu_label": "Diabetes Prediction",
        "title": "Diabetes Prediction Using ML",
        "description": "Structured inputs mirror the engineered features used by the trained diabetes model.",
        "form_key": "diabetes_form",
        "column_count": 3,
        "model_filename": MODEL_FILES["diabetes"],
        "positive_class": 1,
        "idle_message": "Enter the patient details and click Predict.",
        "input_hint": "Binary inputs are encoded for you. Numeric inputs accept validated values only.",
        "messages": {
            "positive": "The person is likely diabetic.",
            "negative": "The person is likely not diabetic.",
        },
        "fields": [
            {"name": "age", "label": "Age", "type": "number", "min_value": 1.0, "max_value": 120.0, "default": 45.0, "step": 1.0},
            {"name": "hypertension", "label": "Hypertension", "type": "binary", "options": {"No": 0, "Yes": 1}},
            {"name": "heart_disease", "label": "Heart disease history", "type": "binary", "options": {"No": 0, "Yes": 1}},
            {"name": "bmi", "label": "BMI", "type": "number", "min_value": 10.0, "max_value": 80.0, "default": 26.5, "step": 0.1},
            {"name": "hba1c_level", "label": "HbA1c level", "type": "number", "min_value": 3.0, "max_value": 15.0, "default": 5.8, "step": 0.1},
            {"name": "blood_glucose_level", "label": "Blood glucose level", "type": "number", "min_value": 50.0, "max_value": 400.0, "default": 110.0, "step": 1.0},
            {"name": "gender_male", "label": "Gender", "type": "binary", "options": {"Female": 0, "Male": 1}},
            {
                "name": "smoking_history",
                "label": "Smoking history",
                "type": "vector_select",
                "feature_names": ["non_smoker", "past_smoker"],
                "options": {
                    "Current smoker": [0, 0],
                    "Non-smoker": [1, 0],
                    "Past smoker": [0, 1],
                },
                "default_index": 1,
            },
        ],
    },
    {
        "menu_label": "Heart Disease Prediction",
        "title": "Heart Disease Prediction Using ML",
        "description": "This screen follows the exact feature order expected by the heart disease model.",
        "form_key": "heart_form",
        "column_count": 3,
        "model_filename": MODEL_FILES["heart"],
        "positive_class": 1,
        "idle_message": "Fill in the cardiac indicators and click Predict.",
        "input_hint": "Use the encoded categories shown in the labels for categorical clinical values.",
        "messages": {
            "positive": "The person is likely to have heart disease.",
            "negative": "The person is likely not to have heart disease.",
        },
        "fields": [
            {"name": "age", "label": "Age", "type": "number", "min_value": 1.0, "max_value": 120.0, "default": 52.0, "step": 1.0},
            {"name": "sex", "label": "Sex", "type": "binary", "options": {"Female (0)": 0, "Male (1)": 1}},
            {"name": "cp", "label": "Chest pain type", "type": "select", "options": {"0": 0, "1": 1, "2": 2, "3": 3}},
            {"name": "trestbps", "label": "Resting blood pressure", "type": "number", "min_value": 50.0, "max_value": 250.0, "default": 125.0, "step": 1.0},
            {"name": "chol", "label": "Serum cholesterol (mg/dl)", "type": "number", "min_value": 100.0, "max_value": 700.0, "default": 212.0, "step": 1.0},
            {"name": "fbs", "label": "Fasting blood sugar > 120 mg/dl", "type": "binary", "options": {"No (0)": 0, "Yes (1)": 1}},
            {"name": "restecg", "label": "Resting ECG result", "type": "select", "options": {"0": 0, "1": 1, "2": 2}},
            {"name": "thalach", "label": "Maximum heart rate achieved", "type": "number", "min_value": 50.0, "max_value": 250.0, "default": 168.0, "step": 1.0},
            {"name": "exang", "label": "Exercise induced angina", "type": "binary", "options": {"No (0)": 0, "Yes (1)": 1}},
            {"name": "oldpeak", "label": "ST depression induced by exercise", "type": "number", "min_value": 0.0, "max_value": 10.0, "default": 1.0, "step": 0.1},
            {"name": "slope", "label": "Slope of peak exercise ST segment", "type": "select", "options": {"0": 0, "1": 1, "2": 2}, "default_index": 2},
            {"name": "ca", "label": "Major vessels colored by fluoroscopy", "type": "select", "options": {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}, "default_index": 2},
            {"name": "thal", "label": "Thal", "type": "select", "options": {"Normal (0)": 0, "Fixed defect (1)": 1, "Reversible defect (2)": 2, "Other (3)": 3}, "default_index": 3},
        ],
    },
    {
        "menu_label": "Parkinsons Prediction",
        "title": "Parkinson's Disease Prediction Using ML",
        "description": "The model expects 22 voice-measurement features from the Parkinson's dataset.",
        "form_key": "parkinsons_form",
        "column_count": 5,
        "model_filename": MODEL_FILES["parkinsons"],
        "positive_class": 1,
        "idle_message": "Enter the voice signal measurements and click Predict.",
        "input_hint": "Default values are based on a sample notebook inference row and can be edited.",
        "messages": {
            "positive": "The person is likely to have Parkinson's disease.",
            "negative": "The person is likely not to have Parkinson's disease.",
        },
        "fields": [
            {"name": "fo", "label": "MDVP:Fo(Hz)", "type": "number", "min_value": 50.0, "max_value": 300.0, "default": 140.341, "step": 0.001},
            {"name": "fhi", "label": "MDVP:Fhi(Hz)", "type": "number", "min_value": 50.0, "max_value": 400.0, "default": 159.774, "step": 0.001},
            {"name": "flo", "label": "MDVP:Flo(Hz)", "type": "number", "min_value": 50.0, "max_value": 300.0, "default": 67.021, "step": 0.001},
            {"name": "jitter_percent", "label": "MDVP:Jitter(%)", "type": "number", "min_value": 0.0, "max_value": 1.0, "default": 0.00817, "step": 0.00001},
            {"name": "jitter_abs", "label": "MDVP:Jitter(Abs)", "type": "number", "min_value": 0.0, "max_value": 1.0, "default": 0.00001, "step": 0.00001},
            {"name": "rap", "label": "MDVP:RAP", "type": "number", "min_value": 0.0, "max_value": 1.0, "default": 0.0043, "step": 0.00001},
            {"name": "ppq", "label": "MDVP:PPQ", "type": "number", "min_value": 0.0, "max_value": 1.0, "default": 0.0044, "step": 0.00001},
            {"name": "ddp", "label": "Jitter:DDP", "type": "number", "min_value": 0.0, "max_value": 1.0, "default": 0.01289, "step": 0.00001},
            {"name": "shimmer", "label": "MDVP:Shimmer", "type": "number", "min_value": 0.0, "max_value": 1.0, "default": 0.03198, "step": 0.00001},
            {"name": "shimmer_db", "label": "MDVP:Shimmer(dB)", "type": "number", "min_value": 0.0, "max_value": 5.0, "default": 0.313, "step": 0.001},
            {"name": "apq3", "label": "Shimmer:APQ3", "type": "number", "min_value": 0.0, "max_value": 1.0, "default": 0.0183, "step": 0.00001},
            {"name": "apq5", "label": "Shimmer:APQ5", "type": "number", "min_value": 0.0, "max_value": 1.0, "default": 0.0181, "step": 0.00001},
            {"name": "apq", "label": "MDVP:APQ", "type": "number", "min_value": 0.0, "max_value": 1.0, "default": 0.02428, "step": 0.00001},
            {"name": "dda", "label": "Shimmer:DDA", "type": "number", "min_value": 0.0, "max_value": 1.0, "default": 0.0549, "step": 0.00001},
            {"name": "nhr", "label": "NHR", "type": "number", "min_value": 0.0, "max_value": 1.0, "default": 0.02183, "step": 0.00001},
            {"name": "hnr", "label": "HNR", "type": "number", "min_value": 0.0, "max_value": 50.0, "default": 19.57, "step": 0.01},
            {"name": "rpde", "label": "RPDE", "type": "number", "min_value": 0.0, "max_value": 2.0, "default": 0.537264, "step": 0.000001},
            {"name": "dfa", "label": "DFA", "type": "number", "min_value": 0.0, "max_value": 2.0, "default": 0.720908, "step": 0.000001},
            {"name": "spread1", "label": "spread1", "type": "number", "min_value": -10.0, "max_value": 5.0, "default": -5.40942, "step": 0.00001},
            {"name": "spread2", "label": "spread2", "type": "number", "min_value": 0.0, "max_value": 2.0, "default": 0.22685, "step": 0.00001},
            {"name": "d2", "label": "D2", "type": "number", "min_value": 0.0, "max_value": 5.0, "default": 2.359973, "step": 0.000001},
            {"name": "ppe", "label": "PPE", "type": "number", "min_value": 0.0, "max_value": 2.0, "default": 0.226156, "step": 0.000001},
        ],
    },
    {
        "menu_label": "Kidney Disease Prediction",
        "title": "Kidney Disease Prediction Using ML",
        "description": "The kidney screen now uses the same six features that the local training script uses.",
        "form_key": "kidney_form",
        "column_count": 3,
        "model_filename": MODEL_FILES["kidney"],
        "positive_class": 0,
        "idle_message": "Enter the kidney-related measurements and click Predict.",
        "input_hint": "Hypertension is encoded with a guided selector to match training data.",
        "messages": {
            "positive": "The person is likely to have kidney disease.",
            "negative": "The person is likely not to have kidney disease.",
        },
        "fields": [
            {"name": "sg", "label": "Specific gravity", "type": "number", "min_value": 1.0, "max_value": 1.05, "default": 1.02, "step": 0.01},
            {"name": "al", "label": "Albumin", "type": "number", "min_value": 0.0, "max_value": 5.0, "default": 0.0, "step": 1.0},
            {"name": "sc", "label": "Serum creatinine", "type": "number", "min_value": 0.0, "max_value": 20.0, "default": 0.7, "step": 0.1},
            {"name": "hemo", "label": "Hemoglobin", "type": "number", "min_value": 0.0, "max_value": 25.0, "default": 13.2, "step": 0.1},
            {"name": "pcv", "label": "Packed cell volume", "type": "number", "min_value": 0.0, "max_value": 60.0, "default": 28.0, "step": 1.0},
            {"name": "htn", "label": "Hypertension", "type": "binary", "options": {"No": 0, "Yes": 1}},
        ],
    },
]
