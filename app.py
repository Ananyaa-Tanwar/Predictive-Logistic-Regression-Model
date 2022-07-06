import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

#Loading the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    #For rendering results on HTML GUI

    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    print(final_features)
    prediction = model.predict(final_features)

    if prediction[0] == 0:
        output = 'Not Hazardous'
    else:
        output = 'Hazardous'

    return render_template('index.html', prediction_text ='The asteroid is {}'.format(output))

# @app.route('predict_api', methods=['POST'])
# def predict_api():
#     #For direct API calls through request

#     data = request.get_json(force=True)
#     prediction = model.predict([np.array(list(data.values()))])

#     output = prediction[0]
#     return jsonify(output)

if __name__ == 'main':
    app.run(debug = True)
