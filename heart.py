from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('a.php')

@app.route('/b', methods = ['POST'])
def run():
	result = request.form
    # Load Model
    with open('model.sav', 'rb') as file :
        model = pickle.load(file)

    # Load Datatest
    df = pd.read_csv('tubes2_HeartDisease_test.csv')

    # Predict
    predict = model.predict(df)

    return render_template('b.php', key = result)

if __name__ == '__main__' :
    app.run(debug = True)
