import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import pickle

model = tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


## streamlit app
st.title("Customer Churn Prediction")


## User input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age",18,92)
balance = st.number_input("Balance")
estimated_salary = st.number_input("Estimated Salary")
credit_score = st.number_input("Credit Score")
tenure = st.number_input("Tenure",0,10)
num_of_products = st.number_input("Number of Products",1,4)    
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

## prediction
prediction = model.predict(input_data_scaled)

prediction_proba = prediction[0][0]

if prediction_proba > 0.5:
    st.write(f"The customer is likely to churn with a probability of {prediction_proba:.2f}")
else:
    st.write(f"The customer is not likely to churn with a probability of {1 - prediction_proba:.2f}")
