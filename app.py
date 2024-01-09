import pickle 
import json
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

#starting point of myapplication
app = Flask(__name__)

#Load the ML model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(data)
        print(np.array(list(data.values())).reshape(1, -1))
        new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
        output = regmodel.predict(new_data)
        print(output[0])
        return jsonify(output[0])
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)})


@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for  x in request.form.values()] #This will capture all the valuesfrom HTML form, try to convert in float because it is better than int for our model.
    final_input= scalar.transform(np.array(data).reshape(1,-1)) #standerdize
    print(final_input)
    output =regmodel.predict(final_input)[0] #value comes in array that's why we added [0]
    return render_template("home.html", prediction_text = f"The House price prediction is {output}")


if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)



