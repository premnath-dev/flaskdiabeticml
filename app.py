from flask import Flask, request, url_for, redirect, render_template
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open("testing5.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    setup_df = pd.DataFrame(pd.Series([int(x) for x in request.form.values()]))
    diabetic_prediction = model.predict_proba(setup_df)
    output = '{:.2%}'.format(diabetic_prediction[0][1])
    output = str(float(output)*100) + "%"
    if output > '50%':
        return render_template("result.html", pred="Diabetic probability is {}".format(output))
    else:
        return render_template("result.html", pred="Non Diabetic")
    

if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=8080)
    app.run(debug=True)
    
