import pandas as pd
import pickle
import streamlit as st

model_path = "disease_prediction_project/Models/"

# Features of the models
diabetes_features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

parkinson_features = ["MDVP:Fo(Hz)","MDVP:Fhi(Hz)","MDVP:Flo(Hz)","MDVP:Jitter(%)","MDVP:Jitter(Abs)","MDVP:RAP","MDVP:PPQ","Jitter:DDP","MDVP:Shimmer","MDVP:Shimmer(dB)","Shimmer:APQ3","Shimmer:APQ5","MDVP:APQ","Shimmer:DDA","NHR","HNR","RPDE","DFA","spread1","spread2","D2","PPE"]

heart_features = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]

# loading the saved model and caching it
@st.cache_resource
def load_models():
    with open(model_path + 'diabetes_trained_model.sav', 'rb') as f:
        diabetes_loaded = pickle.load(f)

    with open(model_path + 'Parkinson_disease_trained_model.sav', 'rb') as f:
        parkinson_loaded=pickle.load(f)

    with open(model_path + 'Heart_disease_trained_model.sav', 'rb') as f:
        heart_disease_loaded=pickle.load(f)

    with open(model_path + 'Parkinson_scaler.sav',"rb") as f:
        scaler = pickle.load(f)
    return diabetes_loaded,parkinson_loaded,heart_disease_loaded,scaler

diabetes_loaded_model,parkinson_loaded_model,heart_disease_loaded_model,scaler = load_models()

def reset_form(prefix):
    keys_to_reset = [key for key in st.session_state if key.startswith(prefix)]
    for key in keys_to_reset:
        if isinstance(st.session_state[key], (int, float)):
            st.session_state[key] = 0
        else:
            st.session_state[key] = ""

def predict(model,input_data,columns,scaler = None):
    df = pd.DataFrame([input_data],columns=columns)

    if scaler:
        df = scaler.transform(df)
    prediction =  model.predict(df)[0]
    probability =  model.predict_proba(df)[0]
    return prediction,probability

def calculate_risk(probability):
    risk_value = probability[1]

    if risk_value < 0.3:
        return "Low Risk",risk_value
    elif risk_value < 0.7:
        return "Moderate Risk", risk_value
    else:
        return "High Risk" , risk_value
    
def show_risk(probability):
    label,value = calculate_risk(probability)

    st.progress(value)

    if label == "Low Risk":
        st.success(label)
    elif label == "Moderate Risk":
        st.warning(label)
    else:
        st.error(label)

def diabetes_prediction(input_data_diabetes):
    try:    
        prediction,probability = predict(diabetes_loaded_model,input_data_diabetes,diabetes_features)
    except Exception as e:
        st.error("Prediction Failed!")
        st.write(e)
        return None
    
    confidence = probability[prediction]*100
    show_risk(probability)
   
    if (prediction == 0):
        return f"Negative! Does not have diabetes. Confidence: {confidence:.2f}%."
    else:
        st.error( f"Positive! Diabetes confirmed. Confidence: {confidence:.2f}%.")
        return None
    
    
def parkinson_disease_prediction(input_data_parkinson):
    try:
        prediction,probability = predict(parkinson_loaded_model,input_data_parkinson,parkinson_features,scaler)
    except Exception as e:
        st.error("Prediction Failed!")
        st.write(e)
        return None
    
    confidence = probability[prediction]*100
    show_risk(probability)

    if (prediction==0):
        return f"The person does not have Parkinson's disease. Confidence: {confidence:.2f}%"
    else:
        st.error (f"The person has Parkinson's disease. Confidence: {confidence:.2f}%")
        return None
    
def heart_disease_prediction(input_data_heart):
    try:
        prediction,probability = predict(heart_disease_loaded_model,input_data_heart,heart_features)
    except Exception as e:
        st.error("Prediction Failed!")
        st.write(e)
        return None
    
    confidence = probability[prediction]*100
    show_risk(probability)
    
    if (prediction==0):
        return f"Heart is healthy! No heart disease. Confidence: {confidence:.2f}%"
    else:
        st.error( f"Person has heart disease. Confidence: {confidence:.2f}%")
        return None
    
def home():
    st.title("Welcome To Multi-Disease Prediction Web App.")
    st.subheader("\n\nPlease Select a disease from sidebar for diagnoses")
    st.write("Disease availabe for diagnoses are:\n1. Diabetes \n2. Parkinson's disease \n3. Heart Disease ")

def Diabetes_button():
    # giving a title
    st.title("Diabetes Prediction")

    # Reset Logic before creating form
    if "reset_diab" in st.session_state and st.session_state["reset_diab"]:
        reset_form("diab_")
        st.session_state["reset_diab"] = False  # reset the flag

    # Form Creation
    with st.form("Diabetes Form"):

        # getting the input data from user
        col = st.columns(4)

        with col[0]:
            Pregnancies = st.number_input("Number of Pregnancies:", min_value=0,step=1,key="diab_preg")
        with col[1]:
            Glucose = st.number_input("Glucose Level:", min_value=0,step=1,key="diab_gluc")
        with col[2]:
            BloodPressure = st.number_input("BloodPressure value:", min_value=0,step=1,key="diab_BP")
        with col[3]:
            SkinThickness = st.number_input("Skin Thickness Value:", min_value=0,step=1,key="diab_ST")

        cols = st.columns(4)
        with cols[0]:
            Insulin = st.number_input("Insulin Level:", min_value=0,step=1,key="diab_Ins")
        with cols[1]:
            BMI = st.number_input("BMI Value:", min_value=0.0,step=0.1,key="diab_bmi")
        with cols[2]:
            DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function:", min_value=0.0,step=0.001,key="diab_pedigfunc")
        with cols[3]:
            Age = st.number_input("Age of the Person:", min_value=0,step=1,key="diab_age")

        submitted = st.form_submit_button("Predict")
    
    # Reset Button
    if st.button("Reset"):
        st.session_state["reset_diab"] = True
        st.rerun()  # rerun the app to apply cleared values

    # prediction
    if submitted:
        if Age < 1:
            st.error("Input Error!Age cannot be 0")
            return None
        with st.spinner("Analysing..."):
            diagnose = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

        if diagnose:
            st.success(diagnose)

def Parkinson_button():
    # giving a title
    st.title("Parkinson's disease Prediction")

    # Reset Logic before creating form
    if "reset_parkinson" in st.session_state and st.session_state["reset_parkinson"]:
        reset_form("park_")
        st.session_state["reset_parkinson"] = False # reset flag

    # Form Creation
    with st.form("Parkinson Form"):
        # getting the input from the user
        cols = st.columns(4)

        with cols[0]:
            MDVP_Fo = st.number_input("MDVP:Fo(Hz) value:", min_value=0.0,key="park_mdfo")
        with cols[1]:
            MDVP_Fhi = st.number_input("MDVP:Fhi(Hz) value:", min_value=0.0,key="park_mdfhi")
        with cols[2]:
            MDVP_Flo = st.number_input("MDVP:Flo(Hz) value:", min_value=0.0,key="park_mdflo")
        with cols[3]:
            MDVP_Jitter = st.number_input("MDVP:Jitter(%) Value:", min_value=0.0,key="park_mdjit")
        
        cols = st.columns(4)

        with cols[0]:
            MDVP_Jitter_abs = st.number_input("MDVP:Jitter(Abs) value:", min_value=0.0,key="park_mdjitabs")
        with cols[1]:
            MDVP_RAP = st.number_input("MDVP:RAP Value:", min_value=0.0,key="park_mdrap")
        with cols[2]:
            MDVP_PPQ = st.number_input("MDVP:PPQ Value:", min_value=0.0,key="park_mdppq")
        with cols[3]:
            Jitter_DDP = st.number_input("Jitter:DDP Value:", min_value=0.0,key="park_jitddp")
        
        cols = st.columns(4)

        with cols[0]:
            MDVP_Shimmer = st.number_input("MDVP:Shimmer Value:", min_value=0.0,key="park_mdshi")
        with cols[1]:
            MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB) Value:", min_value=0.0,key="park_mdshidb")
        with cols[2]:
            Shimmer_APQ3 = st.number_input("Shimmer:APQ3 Value:", min_value=0.0,key="park_shiaq3")
        with cols[3]:
            Shimmer_APQ5 = st.number_input("Shimmer:APQ5 Value:", min_value=0.0,key="park_shiaqq5")

        cols = st.columns(4)

        with cols[0]:
            MDVP_APQ = st.number_input("MDVP:APQ Value:", min_value=0.0,key="park_mdaq")
        with cols[1]:
            Shimmer_DDA = st.number_input("Shimmer:DDA Value:", min_value=0.0,key="park_shidda")
        with cols[2]:
            NHR = st.number_input("NHR Value:", min_value=0.0,key="park_nhr")
        with cols[3]:
            HNR = st.number_input("HNR Value:", min_value=0.0,key="park_hnr")

        cols = st.columns(4)

        with cols[0]:
            RPDE = st.number_input("RPDE Value:", min_value=0.0,key="park_rpde")
        with cols[1]:
            DFA = st.number_input("DFA Value:", min_value=0.0,key="park_dfa")
        with cols[2]:
            spread1 = st.number_input("spread1 Value:", min_value=0.0,key="park_sp1")
        with cols[3]:
            spread2 = st.number_input("spread2 Value:", min_value=0.0,key="park_sp2")

        cols = st.columns(4)

        with cols[0]:
            D2 = st.number_input("D2 Value:", min_value=0.0,key="park_d2")
        with cols[1]:
            PPE = st.number_input("PPE Value:", min_value=0.0,key="park_ppe")

        submitted = st.form_submit_button("Predict")

    # Reset form
    if st.button("Reset"):
        st.session_state["reset_parkinson"] = True
        st.rerun() # rerun the app to apply cleared values

    # prediction

    if submitted:
        with st.spinner("Analysing..."):
            parkinson_diagnose = parkinson_disease_prediction([MDVP_Fo,MDVP_Fhi,MDVP_Flo,MDVP_Jitter,MDVP_Jitter_abs,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])
        if parkinson_diagnose:
            st.success(parkinson_diagnose)

def heart_button():
    # giving a title
    st.title("Heart Disease Prediction")

    # Reset before creating form
    if "reset_heart" in st.session_state and st.session_state["reset_heart"]:
        reset_form("heart_")
        st.session_state["reset_heart"] = False #reset flag
    
    # Form Creation
    with st.form("Heart Disease Form"):
        # getting the input from the user
        cols = st.columns(4)

        with cols[0]:
            age = st.number_input("Age of the person:", min_value=0,key="heart_age")
        with cols[1]:
            sex = st.selectbox("Sex", ["Female", "Male"])            
            sex_map = {"Female": 0 , "Male":1}
            sex = sex_map[sex]
        with cols[2]:
            cp = st.number_input("cp value:", min_value=0,key="heart_cp")
        with cols[3]:
            trestbps = st.number_input("trestbps Value:", min_value=0,key="heart_tres")

        cols = st.columns(4)

        with cols[0]:
            chol = st.number_input("chol Value:", min_value=0,key="heart_chol")
        with cols[1]:
            fbs = st.number_input("fbs Value:", min_value=0,key="heart_fbs")
        with cols[2]:
            restecg = st.number_input("restecg Value:", min_value=0,key="heart_rest")
        with cols[3]:
            thalach = st.number_input("thalach Value:", min_value=0,key="heart_thal")

        cols = st.columns(4)

        with cols[0]:
            exang = st.number_input("exang Value:", min_value=0,key="heart_exang")
        with cols[1]:
            oldpeak = st.number_input("oldpeak Value:", min_value=0,key="heart_oldp")
        with cols[2]:
            slope = st.number_input("slope Value:", min_value=0,key="heart_slp")
        with cols[3]:
            ca = st.number_input("ca Value:", min_value=0,key="heart_ca")
        
        cols = st.columns(4)

        with cols[0]:
            thal = st.number_input("thal Value:", min_value=0,key="heart_th")

        submitted = st.form_submit_button("Predict")

    # Reset Form
    if st.button("Reset"):
        st.session_state["reset_heart"] = True
        st.rerun() # rerun the app to apply cleared values

    # prediction

    if submitted:
        if age < 1:
                st.error("Input Error! Age Cannot be 0.")
                return
        with st.spinner("Analysing..."):
            heart_diagnose = heart_disease_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])

        if heart_diagnose:
            st.success(heart_diagnose)

def main():

    page = st.sidebar.selectbox('Select Disease',["Home","Diabetes", "Parkinson","Heart Disease"])

    # opening the selected page
    if page == "Home":
        home()
    elif page == "Diabetes":
        Diabetes_button()
    elif page == "Parkinson":
        Parkinson_button()
    elif page == "Heart Disease":
        heart_button()

if __name__ == '__main__':
    main()



# to run, enter in terminal: 
# streamlit run "C:\Users\shahg\code\disease_prediction_project\diabetes_prediction_web_app.py"