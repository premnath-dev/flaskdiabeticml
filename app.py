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
    inputs = [request.form[str(i)] for i in range(1, 9)]
    inputs = [str(x) for x in inputs]
    setup_df = pd.DataFrame([inputs])
    diabetic_prediction = model.predict_proba(setup_df)
    output = diabetic_prediction[0][1]  
    output_percentage = f"{output:.2%}"
    if output > 0.5:
        return render_template("result.html", pred="Diabetic probability is {}".format(output_percentage))
    else:
        return render_template("result.html", pred="Non Diabetic")
    

if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=8080)
    app.run(debug=True)
    
