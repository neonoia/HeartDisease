from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('AI.html')

@app.route('/result', methods = ['POST'])
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

    # Normalize Data
    rest_blood = ((float(rest_blood)*2)-200)/200
    serum_chol = ((float(serum_chol)*2)-529)/529
    max_heart_rate = ((float(max_heart_rate)*2)-202-60)/(202-60)
    depression = ((float(depression)*2)-62+2.6)/(62+2.6)

    # Create data
    if (rest_ecg == 0) :
        if (peak == 1) :
            data = np.array([[int(age),int(sex),int(chestpain),rest_blood,serum_chol,float(fast_blood),max_heart_rate,float(exercise),depression,1,0,0,1,0,0]])
        elif (peak == 2) :
            data = np.array([[int(age),int(sex),int(chestpain),rest_blood,serum_chol,float(fast_blood),max_heart_rate,float(exercise),depression,1,0,0,0,1,0]])
        else :
            data = np.array([[int(age),int(sex),int(chestpain),rest_blood,serum_chol,float(fast_blood),max_heart_rate,float(exercise),depression,1,0,0,0,0,1]])
    elif (rest_ecg == 1) :
        if (peak == 1) :
            data = np.array([[int(age),int(sex),int(chestpain),rest_blood,serum_chol,float(fast_blood),max_heart_rate,float(exercise),depression,0,1,0,1,0,0]])
        elif (peak == 2) :
            data = np.array([[int(age),int(sex),int(chestpain),rest_blood,serum_chol,float(fast_blood),max_heart_rate,float(exercise),depression,0,1,0,0,1,0]])
        else :
            data = np.array([[int(age),int(sex),int(chestpain),rest_blood,serum_chol,float(fast_blood),max_heart_rate,float(exercise),depression,0,1,0,0,0,1]])
    else :
        if (peak == 1) :
            data = np.array([[int(age),int(sex),int(chestpain),rest_blood,serum_chol,float(fast_blood),max_heart_rate,float(exercise),depression,0,0,1,1,0,0]])
        elif (peak == 2) :
            data = np.array([[int(age),int(sex),int(chestpain),rest_blood,serum_chol,float(fast_blood),max_heart_rate,float(exercise),depression,0,0,1,0,1,0]])
        else :
            data = np.array([[int(age),int(sex),int(chestpain),rest_blood,serum_chol,float(fast_blood),max_heart_rate,float(exercise),depression,0,0,1,0,0,1]])

    # Predict
    predict = model.predict(data)

    return render_template('result.html', result = predict[0])

if __name__ == '__main__' :
    app.run(debug = True)
