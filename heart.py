from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('AI.html')

@app.route('/', methods = ['POST'])
def run():
    # Get data input
    name = request.form['fname'] + request.form['lname']
    age = request.form['age']
    sex = request.form['sex']
    chestpain = request.form['chestpain_type']
    rest_blood = request.form['resting_blood_pressure']
    serum_chol = request.form['serum_cholesterol']
    fast_blood = request.form['sugar']
    rest_ecg = request.form['resting']
    max_heart_rate = request.form['max_heart_rate']
    exercise = request.form['include_angina']
    depression = request.form['depression']
    peak = request.form['peak_exercise']

    # Load Model
    with open('model.sav', 'rb') as file :
        model = pickle.load(file)

    # Create data
    if (rest_ecg == 0) :
        if (peak == 1) :
            data = np.array([[int(age),int(sex),int(chestpain),float(rest_blood),float(serum_chol),float(fast_blood),float(max_heart_rate),float(exercise),float(depression),1,0,0,1,0,0]])
        elif (peak == 2) :
            data = np.array([[int(age),int(sex),int(chestpain),float(rest_blood),float(serum_chol),float(fast_blood),float(max_heart_rate),float(exercise),float(depression),1,0,0,0,1,0]])
        else :
            data = np.array([[int(age),int(sex),int(chestpain),float(rest_blood),float(serum_chol),float(fast_blood),float(max_heart_rate),float(exercise),float(depression),1,0,0,0,0,1]])
    elif (rest_ecg == 1) :
        if (peak == 1) :
            data = np.array([[int(age),int(sex),int(chestpain),float(rest_blood),float(serum_chol),float(fast_blood),float(max_heart_rate),float(exercise),float(depression),0,1,0,1,0,0]])
        elif (peak == 2) :
            data = np.array([[int(age),int(sex),int(chestpain),float(rest_blood),float(serum_chol),float(fast_blood),float(max_heart_rate),float(exercise),float(depression),0,1,0,0,1,0]])
        else :
            data = np.array([[int(age),int(sex),int(chestpain),float(rest_blood),float(serum_chol),float(fast_blood),float(max_heart_rate),float(exercise),float(depression),0,1,0,0,0,1]])
    else :
        if (peak == 1) :
            data = np.array([[int(age),int(sex),int(chestpain),float(rest_blood),float(serum_chol),float(fast_blood),float(max_heart_rate),float(exercise),float(depression),0,0,1,1,0,0]])
        elif (peak == 2) :
            data = np.array([[int(age),int(sex),int(chestpain),float(rest_blood),float(serum_chol),float(fast_blood),float(max_heart_rate),float(exercise),float(depression),0,0,1,0,1,0]])
        else :
            data = np.array([[int(age),int(sex),int(chestpain),float(rest_blood),float(serum_chol),float(fast_blood),float(max_heart_rate),float(exercise),float(depression),0,0,1,0,0,1]])

    # Normalize Data
    #data = StandardScaler().fit_transform(data)
    #data[0] = StandardScaler().fit_transform(data[0])
    #data[0] = StandardScaler().fit_transform(data[0])
    #data[0] = StandardScaler().fit_transform(data[0])
    #rest_blood = StandardScaler().fit_transform(float(rest_blood))
    #serum_chol = StandardScaler().fit_transform(float(serum_chol))
    #max_heart_rate = StandardScaler().fit_transform(float(max_heart_rate))
    #depression = StandardScaler().fit_transform(float(depression))

    # Predict
    predict = model.predict(data)

    return render_template('AI.html', result = predict[0])

if __name__ == '__main__' :
    app.run(debug = True)
