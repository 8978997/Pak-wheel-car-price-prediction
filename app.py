from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('model_car_price_predict.pkl', 'rb'))
label_encoders = pickle.load(open('label_encoders.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    KM_Driven = int(request.form['KM_Driven'])
    CC = int(request.form['CC'])
    Model = int(request.form['Model'])
    Engine_type = request.form['Engine_type']
    CarName = request.form['CarName']
    Transmission = request.form['Transmission']

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
        le = label_encoders[col]
        if new_data[col][0] not in le.classes_:
            le.classes_ = np.append(le.classes_, new_data[col][0])
        new_data[col] = le.transform(new_data[col])

    numerical_cols = ['Model', 'CC', 'Km_Driven']
    new_data[numerical_cols] = scaler.transform(new_data[numerical_cols])

    new_data = new_data[['Model', 'CC', 'Engine_type', 'Transmission', 'Km_Driven', 'CarName']]

    prediction = model.predict(new_data)

    return render_template('after.html', prediction=round(prediction[0]), )


if __name__ == '__main__':
    app.run(debug=True)

