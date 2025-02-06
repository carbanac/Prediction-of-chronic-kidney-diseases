from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictions', methods=['GET', 'POST'])
def predictions():
    if request.method == 'POST':
        # Get form data
        data = request.form
        features = np.array([
            data['age'],
            data['blood_pressure'],
            data['specific_gravity'],
            data['albumin'],
            data['sugar'],
            1 if data['red_blood_cells'] == 'normal' else 0,
            1 if data['pus_cell'] == 'normal' else 0,
            1 if data['pus_cell_clumps'] == 'present' else 0,
            1 if data['bacteria'] == 'present' else 0,
            data['blood_glucose_random'],
            data['blood_urea'],
            data['serum_creatinine'],
            data['sodium'],
            data['potassium'],
            data['haemoglobin'],
            data['packed_cell_volume'],
            data['white_blood_cell_count'],
            data['red_blood_cell_count'],
            1 if data['hypertension'] == 'yes' else 0,
            1 if data['diabetes_mellitus'] == 'yes' else 0,
            1 if data['coronary_artery_disease'] == 'yes' else 0,
            1 if data['appetite'] == 'good' else 0,
            1 if data['peda_edema'] == 'yes' else 0,
            1 if data['aanemia'] == 'yes' else 0
        ], dtype=float).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(features)[0]
        result = 'Chronic Kidney Disease' if prediction == 0 else 'No Chronic Kidney Disease'

        return render_template('result.html', result=result)
    return render_template('predictions.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)
