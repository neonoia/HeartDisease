from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('a.php')

@app.route('/b', methods = ['POST'])
def run():
    result = request.form['Mathematics']

    # Load Model
    with open('model.sav', 'rb') as file :
        model = pickle.load(file)

    # Create dataform
    data = np.array([[54,1,4,-0.044575,0.431818,0.0,-1.784928,1.0,-0.324500,1,0,0,0,1,0]])

    # Predict
    predict = model.predict(data)

    return render_template('b.php', result = predict[0])

if __name__ == '__main__' :
    app.run(debug = True)
