import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/DepartmentalStore')
def dmart():
    return render_template('Dmart.html')

@app.route('/Supermarket')
def BigBazaar():
    return render_template('BigBazaar.html')

@app.route('/AllInOne')
def Reliance():
    return render_template('Reliance.html')

@app.route('/EWay')
def croma():
    return render_template('croma.html')

@app.route('/AlwaysConnected')
def jiomart():
    return render_template('jiomart.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 0)

    return render_template('index.html', prediction_text='Expected crowd should be  {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)