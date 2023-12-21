from PIL import Image
import os
import io
import shap
import pandas as pd
import base64
import matplotlib.pyplot as plt
import joblib
import imblearn
import numpy as np
import streamlit as st

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

def main():
    # st.set_page_config(layout="wide")
    st.title("AMI Mortality Prediction")
    st.write("Enter the following indicators to predict survival probability.")

    col1, col2, col3 = st.columns([0.5, 0.1, 0.5])
    with col1:
        st.subheader("Index Data: ")
        # Input fields
        age = st.number_input("Age (18-90 years)", value=89, min_value=18, max_value=90, step=1)
        bmi = st.number_input("BMI (13.81-44.67)", value=26.81, min_value=13.81, max_value=44.67, step=0.01)
        apsiii = st.number_input("APSIII (0-108)", value=56, min_value=0, max_value=108, step=1)
        glucose = st.number_input("Glucose (10-280 mg/dL)", value=198.00, min_value=10.00, max_value=280.00, step=0.01)
        hr = st.number_input("HR (39.0-130.0 bpm)", value=66.0, min_value=39.0, max_value=130.0, step=0.1)
        rr = st.number_input("RR (3.0-35.0 insp/bpm)", value=29.0, min_value=3.0, max_value=35.0, step=0.1)
        sysbp = st.number_input("SysBP (43.0-198.0 mmHg)", value=159.0, min_value=43.0, max_value=198.0, step=0.1)
        
    with col3:
        st.subheader("...")
        ast = st.number_input("AST (3.0-217.0 IU/L)", value=66.0, min_value=2.0, max_value=217.0, step=0.1)
        troponint = st.number_input("Troponin_T (0.01-7.27 ng/mL)", value=6.00, min_value=0.01, max_value=7.27, step=0.01)
        hemoglobin = st.number_input("Hemoglobin (4.2-17.2 g/dL)", value=14.2, min_value=4.2, max_value=17.2, step=0.1)
        ag = st.number_input("AG (3.0-27.0 mEq/L)", value=18.0, min_value=3.0, max_value=27.0, step=0.1)
        ca = st.number_input("Total_Ca (6.6-10.1 mg/dL)", value=7.6, min_value=6.6, max_value=10.1, step=0.1)
        egfr = st.number_input("eGFR (1.36-184.11)", value=22.3, min_value=1.36, max_value=184.11, step=0.01)
        surgery_sign = st.checkbox("Has surgery?")
        drug_sign = st.checkbox("Take drug? ")

        if surgery_sign:
            surgery = 1 
        else: surgery = 0
        if drug_sign:
            drug = 1 
        else: drug = 0


    st.markdown('---')
    col1, col2 = st.columns([0.3, 0.7])

    with col1:
        st.subheader("Your Data:")
        single_data = [age, egfr, apsiii, glucose, surgery,
                       drug, bmi, ast, troponint, hemoglobin,
                       ag, hr, rr, sysbp, ca]
        numerical_encoded_data = [float(x) for x in single_data]

        pd_data = pd.DataFrame(
            np.array(numerical_encoded_data).reshape(1, -1),
            columns=['Age', 'eGFR', 'APSIII', 'Glucose', 'Surgery', 'Drug', 'BMI', 'AST',
                     'Troponin_T', 'Hemoglobin', 'AG', 'HR', 'RR', 'SysBP', 'Total_Ca']
        )

        json_data = {
            'Age': age,
            'eGFR': egfr,
            'APSIII': apsiii,
            'Glucose': glucose,
            'Surgery': surgery,
            'Drug': drug,
            'BMI': bmi,
            'AST': ast,
            'Troponin_T': troponint,
            'Hemoglobin': hemoglobin,
            'AG': ag,
            'HR': hr,
            'RR': rr,
            'SysBP': sysbp,
            'Total_Ca': ca,
            'Surgery' : surgery,
            'Drug': drug
        }

        st.write(json_data)

    with col2:
        # Load Model 1 and calculate survival probability
        model = load_model('models/BRF_mimic.pkl')
        prediction = model.predict(np.array(numerical_encoded_data).reshape(1, -1))
        prediction_label = {"Die": 1, "Live": 2}
        final_result = get_key(prediction[0], prediction_label)
        pred_prob = model.predict_proba(np.array(numerical_encoded_data).reshape(1, -1))
        pred_probability_score = {
            "Live Probability": round(pred_prob[0][0] * 100, 1),
            "Die Probability": round(pred_prob[0][1] * 100, 1)
        }

        st.subheader("Prediction")
        # st.write("Survival Probability:", final_result)
        # st.write("Probability Scores:", pred_probability_score)
        st.metric("Live Probability Scores:", 
            pred_probability_score['Live Probability'], 
            pred_probability_score['Die Probability'])

        # Load Model 2 and generate waterfall plot
        explainer = load_model('models/shap_explainer.pkl')

        waterfall_shap = explainer(pd_data)[:, :, 1]
        shap.plots.waterfall(waterfall_shap[0], max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)

        # Convert the plot to an image
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image = Image.open(buffer)

        # Display the waterfall plot
        st.subheader("SHAP Waterfall Plot")
        st.image(image, use_column_width=True)

if __name__ == "__main__":
    main()