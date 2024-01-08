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


if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)



