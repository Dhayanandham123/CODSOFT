# app.py - Flask Web Application

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open('model.pkl', 'rb'))
le_sex = pickle.load(open('le_sex.pkl', 'rb'))
le_embarked = pickle.load(open('le_embarked.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = int(request.form['pclass'])
        sex = le_sex.transform([request.form['sex']])[0]
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        embarked = le_embarked.transform([request.form['embarked']])[0]

        features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
        prediction = model.predict(features)[0]

        result = "Survived" if prediction == 1 else "Did not survive"
        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
