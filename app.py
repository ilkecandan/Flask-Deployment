"""FLASK_APP main.py for prediction of Titanic Survivors
"""

from pathlib import Path
import pickle
from flask import Flask, request
from flask import render_template
app = Flask(__name__)


def load_model():
    """Loading the mode to file"""
    script_location = Path(__file__)
    prediction_model = pickle.load(open((script_location / '../model.pkl'), 'rb'))
    return prediction_model


def prediction(values):
    prediction_model = load_model()
    predict_results = prediction_model.predict([values])
    return predict_results[0]

@app.route('/')
def index():
    """Landing Page for app.py"""
    return render_template('index.html')


@app.route('/predict', methods = ['POST'])
def get_prediction():
    """Landing Page loading"""
    # age = int(request.form.get('age'))
    #load the form data to variables
    siblings = int(request.form.get('siblings'))
    parents = int(request.form.get('parents'))
    if request.form.get('gender') == 'Male':
        gender = 1
    else:
        gender = 0
    #call the prediction function utilizing the form variables/features
    #Pclass feature hard-coded as '2'
    predict_results = prediction([gender, siblings, parents, 2])
    #set predict value to True
    # To enable Jinja code on the HTML to enable view of the results section.
    predict = True
    #render index.html & pass relevant variables to be displayed.
    return render_template('index.html',
                            predict = predict,
                            predict_results = predict_results,
                            age = siblings )
# initiate app & load model to memory
if __name__ == '__main__':
    load_model()
    app.run()