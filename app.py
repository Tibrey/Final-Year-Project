from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

from logistic import LogisticRegression
from SVM import SVM

# from Logistic_Regression.ipynb import LogisticRegression

app = Flask(__name__, static_url_path="/static")

# Load the trained model from the .pkl file
with open("../model.pkl", "rb") as file:
    model = LogisticRegression.load_model("../model.pkl")

# with open("../SVM_model.pkl", "rb") as file:
#     model = SVM.load_model("../SVM_model.pkl")

# print(model)


# model = pickle.load(open("../model.pkl","rb"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    # if request.method == 'POST':
    #     try:
    #         # Assuming your form has input fields with names like 'feature1', 'feature2', etc.
    #         features = [float(x) for x in request.form.values()]
    #         final_features = np.array([features])

    #         # Make a prediction using the model
    #         prediction = model.predict(final_features)

    #         # Format the prediction for display
    #         formatted_prediction = "The predicted value is {}".format(prediction[0])

    #         return render_template('predict.html', prediction=formatted_prediction)
    #     except Exception as e:
    #         error_message = "An error occurred: {}".format(str(e))
    #         return render_template('predict.html', prediction=error_message)

    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        print(request)
        print(features)
        final_features = [np.array(features)]

        # print(final_features)
        # # final_features = scaler.transform(final_features)
        # features_name = [
        #     "texture_mean",
        #     "area_mean",
        #     "concavity_mean",
        #     "area_se",
        #     "concavity_se",
        #     "fractal_dimension_se",
        #     "smoothness_se",
        #     "concavity_worst",
        #     "symmetry_worst",
        #     "fractal_dimension_worst",
        # ]
        # df = pd.DataFrame(final_features, columns=features_name)
        # print(df.values())
        output = model.predict(final_features)
        # prediction = model.predict(final_features)
        # y_probabilities_test = model.predict_proba(final_features)
        # y_prob_success = y_probabilities_test[:, 1]
        # print("final features", final_features)
        # print("prediction:", prediction)
        # output = round(prediction[0], 2)
        # y_prob = round(y_prob_success[0], 3)
        print(output)

        # if output == 0:
        #     return render_template(
        #         "predict.html",
        #         prediction="THE PATIENT IS MORE LIKELY TO HAVE A BENIGN CANCER WITH PROBABILITY VALUE  {}".format(
        #             y_prob
        #         ),
        #     )
        # else:
        #     return render_template(
        #         "predict.html",
        #         prediction="THE PATIENT IS MORE LIKELY TO HAVE A MALIGNANT CANCER WITH PROBABILITY VALUE  {}".format(
        #             y_prob
        #         ),
        #     )
        if output == 1:
            res_val = "Benign"
        # elif output == 0:
        #     res_val = "Malignant"
        else:
            res_val = "Malignant"

        return render_template(
            "predict.html", prediction="Patient has {}".format(res_val)
        )

    elif request.method == "GET":
        return render_template("predict.html")


@app.route("/data")
def data():
    return render_template("data.html")


@app.route("/faq")
def faq():
    return render_template("faq.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)
