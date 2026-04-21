import os
import pickle
import streamlit as st
import time
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="MEDCHECK",
                   layout="wide",
                   page_icon="🧑‍⚕️"
                   
                   )
# st.markdown("""
# <style>
# 	[data-testid="stDecoration"] {
# 		display: none;
# 	}

# </style>""",
# unsafe_allow_html=True)

st.markdown("""
<style>
    [data-testid="stDecoration"] {
        background: linear-gradient(to right, #0070f3, #ffffff);
        background-size: cover;
        background-position: center;
        display: block;
    }
    div.stButton > button:first-child {
        color: lightblue;
    }
    div.stButton > button:hover {
        color: blue;
        background-color: lightblue;
        border-color: blue;
    }
    div.stButton > button:active {
        color: blue;
        border-color: blue;
    }
    div.stButton > button:focus {
        border-color: blue !important;
        background-color: lightblue !important;
        color: blue !important;
        box-shadow: none !important;
    }
</style>        
""", unsafe_allow_html=True)


# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/src/diabetes_prediction_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/src/heart_prediction_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/src/parkinsons_prediction_model.sav', 'rb'))

kidney_model = pickle.load(open(f'{working_dir}/src/kidney_prediction_model.sav', 'rb'))
# sidebar for navigation
with st.sidebar:
    selected = option_menu('MEDCHECK',

                           ['Home Page',
                            'Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Kidney Disease Prediction'],
                           menu_icon='hospital-fill',
                           icons=['house','activity', 'heart', 'person', 'droplet'],
                           default_index=0,
                           styles={
                               "nav-link-selected": {"background-color": "blue"}
                           })

# Diabetes Prediction Page
if selected == 'Home Page':

    # page title
    st.markdown("<h1 style='color: lightblue;'>MEDCHECK - Multiple Disease Prediction System</h1>", unsafe_allow_html=True)

    st.markdown("<p style='font-size: 18px;'>Welcome to MEDCHECK, your personalized health companion. Utilizing advanced machine learning algorithms, MEDCHECK empowers you to predict multiple diseases with accuracy. Take control of your health today!</p>", unsafe_allow_html=True)

    # Add a GIF stored locally to the page
    st.image("MedCheck/home.gif")
    

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.markdown("<h1 style='color: lightblue;'>Diabetes Prediction Using ML</h1>", unsafe_allow_html=True)
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1: 
        Age = st.text_input('Age of the Person')

    with col2:
        Hypertension = st.text_input('Person have Hypertension?')

    with col3:
        HeartDisease = st.text_input('Have Heart Disease?')

    with col1:
        Bmi = st.text_input('BMI Value')

    with col2:
        Hemoglobin = st.text_input('Hemoglobin A1C Level')

    with col3:
        BloodGlusoce = st.text_input('Blood Glucose Level')

    with col1:
        Gender = st.text_input('Is Person a Male?')

    with col2:
        NonSmoker = st.text_input('Is Person a Non-Smoker?')

    with col3:
        PastSmoker = st.text_input('Is Person a Past-Smoker?')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button(label="Click to see Result", type="secondary"):
        with st.spinner('Wait for it...'):
            time.sleep(5)

        user_input = [Age, Hypertension, HeartDisease, Bmi, Hemoglobin,
                      BloodGlusoce, Gender, NonSmoker, PastSmoker]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.info(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.markdown("<h1 style='color: lightblue;'>Heart Disease Prediction Using ML</h1>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex; 1 = male; 0 = female')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button(label="Click to see Result", type="secondary"):
        with st.spinner('Wait for it...'):
            time.sleep(5)

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.info(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.markdown("<h1 style='color: lightblue;'>Parkinson's Disease Prediction Using ML</h1>", unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP- Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP- Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP- Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP- Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP- Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP- RAP')

    with col2:
        PPQ = st.text_input('MDVP- PPQ')

    with col3:
        DDP = st.text_input('Jitter- DDP')

    with col4:
        Shimmer = st.text_input('MDVP- Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP- Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer- APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer- APQ5')

    with col3:
        APQ = st.text_input('MDVP- APQ')

    with col4:
        DDA = st.text_input('Shimmer- DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button(label="Click to see Result", type="secondary"):
        with st.spinner('Wait for it...'):
            time.sleep(5)

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.info(parkinsons_diagnosis)

    # Kidney Prediction Page
if selected == 'Kidney Disease Prediction':

    # page title
    st.markdown("<h1 style='color: lightblue;'>Kidney Disease Prediction Using ML</h1>", unsafe_allow_html=True)
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1: 
        SG = st.text_input('Specific Gravity Value')

    with col2:
        AL = st.text_input('Albumin Value')

    with col3:
        SC = st.text_input('Serum Creatinine Value')

    with col1:
        HB = st.text_input('Hemoglobin Level')

    with col2:
        PCV = st.text_input('Packed Cell Volume Value')

    with col3:
        HT = st.text_input('Hypertension')

    # code for Prediction
    kidney_diagnosis = ''

    # creating a button for Prediction
    if st.button(label="Click to see Result", type="secondary"):
        with st.spinner('Wait for it...'):
            time.sleep(5)

        user_input = [SG, AL, SC, HB, PCV, HT]

        user_input = [float(x) for x in user_input]

        kidney_prediction = kidney_model.predict([user_input])

        if kidney_prediction[0] == 1:
            kidney_diagnosis = 'The person have kidney disease'
        else:
            kidney_diagnosis = 'The person does not have kidney disease'

    st.info(kidney_diagnosis)
