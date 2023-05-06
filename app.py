from flask import Flask, render_template, request, jsonify
import joblib
import json

app = Flask(__name__)

model = joblib.load('lastmymodelRandomForestClassifier.h5')
scaler = joblib.load('lastmyscalerRandomForestClassifier.h5')

@app.route('/', methods=['GET'])
def home():
    return render_template('/Home.html')

@app.route('/index', methods=['GET'])
def pro():
    return render_template('/index.html')

@app.route('/predictAPI', methods=['POST'])
def API():
    data = request.data
    data_dict = json.loads(data)
    gender = int(data_dict['gender'])
    age = int(data_dict['age'])
    cigarettes = int(data_dict['cigarettes'])
    is_high_blood_pressure = int(data_dict['is_high_blood_pressure'])
    blood_pressure_med_treatment = int(data_dict['blood_pressure_med_treatment'])
    is_diabetes = int(data_dict['is_diabetes'])
    total_cholesterol = float(data_dict['total_cholesterol'])
    systolic_blood_pressure = float(data_dict['systolic_blood_pressure'])
    Diastolic_b_p = float(data_dict['Diastolic_b_p'])
    glucose = float(data_dict['glucose'])

    predict_disease = model.predict(scaler.transform([[gender, age, cigarettes, is_high_blood_pressure, is_diabetes, blood_pressure_med_treatment, total_cholesterol, systolic_blood_pressure, Diastolic_b_p, glucose]]))
    predict_disease_a = predict_disease.tolist()

    return jsonify({'predict_disease': predict_disease_a})

@app.route('/predict', methods=['GET'])
def predict():
    gender = int(request.args.get('gender'))
    age = int(request.args.get('age'))
    cigarettes = int(request.args.get('cigarettes'))
    is_high_blood_pressure = int(request.args.get('is_high_blood_pressure'))
    blood_pressure_med_treatment = int(request.args.get('blood_pressure_med_treatment'))
    is_diabetes = int(request.args.get('is_diabetes'))
    total_cholesterol = float(request.args.get('total_cholesterol'))
    systolic_blood_pressure = float(request.args.get('systolic_blood_pressure'))
    Diastolic_b_p = float(request.args.get('Diastolic_b_p'))
    glucose = float(request.args.get('glucose'))

    predict_disease = model.predict(scaler.transform([[gender, age, cigarettes, is_high_blood_pressure, is_diabetes, blood_pressure_med_treatment, total_cholesterol, systolic_blood_pressure, Diastolic_b_p, glucose]]))

    return render_template('/predict.html', predict_disease=predict_disease)

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1')
