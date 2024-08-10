import streamlit as st
import pickle
import pandas as pd
import numpy as np

@st.cache_resource
def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

# Load the model, label encoders, and scaler
model = load_pickle('model_car_price_predict.pkl')
label_encoders = load_pickle('label_encoders.pkl')
scaler = load_pickle('scaler.pkl')

st.title('Car Price Prediction')

st.write("""
## Note:
1. This model predicts the price of a car based on the provided inputs.
2. The current accuracy of the model is 88%, with potential for further improvement.
3. Example input: KM Driven: 3000, CC: 1800, Engine Type: Petrol, Transmission: Automatic, Car Name: Honda Civic
""")

# Inputs
KM_Driven = st.number_input('KM Driven', min_value=0, value=0)
CC = st.number_input('CC', min_value=0, value=0)
Model = st.number_input('Model', min_value=0, value=0)
Engine_type = st.text_input('Engine Type')
CarName = st.text_input('Car Name')
Transmission = st.text_input('Transmission')

if st.button('Predict'):
    new_data1 = {
        'Transmission': Transmission,
        'Engine_type': Engine_type,
        'CarName': CarName,
        'Model': Model,
        'CC': CC,
        'Km_Driven': KM_Driven
    }

    new_data = pd.DataFrame(new_data1, index=[0])

    categorical_cols = ['Engine_type', 'Transmission', 'CarName']
    for col in categorical_cols:
        le = label_encoders.get(col)
        if le is not None:
            if new_data[col][0] not in le.classes_:
                le.classes_ = np.append(le.classes_, new_data[col][0])
            new_data[col] = le.transform(new_data[col])
        else:
            st.error(f"Label encoder for {col} not found.")

    numerical_cols = ['Model', 'CC', 'Km_Driven']
    if set(numerical_cols).issubset(new_data.columns):
        new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])
    else:
        st.error("Numerical columns missing or not in the expected format.")

    new_data = new_data[['Model', 'CC', 'Engine_type', 'Transmission', 'Km_Driven', 'CarName']]

    try:
        prediction = model.predict(new_data)
        st.write(f"The predicted price of the car is: {round(prediction[0], 2)} lacs")
    except Exception as e:
        st.error(f"Prediction error: {e}")
