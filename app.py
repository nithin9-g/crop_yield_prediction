from flask import Flask,request, render_template
import os
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)

# loading models using paths relative to this file
BASE_DIR = os.path.dirname(__file__)
try:
    dtr_path = os.path.join(BASE_DIR, 'dtr.pkl')
    dtr = pickle.load(open(dtr_path, 'rb'))
except FileNotFoundError:
    raise FileNotFoundError(f"dtr.pkl not found at {dtr_path}")

try:
    # the notebook saved the preprocessor with the filename 'preprocesser.pkl'
    preproc_path = os.path.join(BASE_DIR, 'preprocesser.pkl')
    preprocessor = pickle.load(open(preproc_path, 'rb'))
except FileNotFoundError:
    raise FileNotFoundError(f"preprocesser.pkl not found at {preproc_path}")

#flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route("/predict",methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        Area = request.form['Area']
        Item  = request.form['Item']

        features = np.array([[Year,average_rain_fall_mm_per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        transformed_features = preprocessor.transform(features)
        prediction = dtr.predict(transformed_features).reshape(1,-1)

        return render_template('index.html',prediction = prediction[0][0])

if __name__=="__main__":
    app.run(debug=True)