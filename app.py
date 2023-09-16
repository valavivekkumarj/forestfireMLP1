import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


# import pickle models:
ridge_model = pickle.load(
    open("Deployement ML P1\models\Ridgemodel.pkl", "rb")
)  # ridge model
scaller_model = pickle.load(
    open("Deployement ML P1\models\scallermodelforestfire.pkl", "rb")
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["POST", "GET"])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get("Temperature"))
        RH = float(request.form.get("RH"))
        Ws = float(request.form.get("Ws"))
        Rain = float(request.form.get("Rain"))
        FFMC = float(request.form.get("FFMC"))
        DMC = float(request.form.get("DMC"))
        ISI = float(request.form.get("ISI"))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get("Region"))

        scalled_data = scaller_model.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )

        result = ridge_model.predict(scalled_data)

        return render_template("home.html", result=result[0])

    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
