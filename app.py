from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
import sklearn.linear_model

app = Flask(__name__)

# --- Custom Unpickler to handle old LogisticRegression path ---
class FixOldLogisticRegressionUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "sklearn.linear_model.logistic" and name == "LogisticRegression":
            return sklearn.linear_model.LogisticRegression
        return super().find_class(module, name)

# --- Load the model using the custom unpickler ---
with open("model.pkl", "rb") as f:
    model = FixOldLogisticRegressionUnpickler(f).load()


@app.route('/')
def home():
    return render_template("forest.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_features = [int(x) for x in request.form.values()]
        final = [np.array(int_features)]
        prediction = model.predict_proba(final)
        output = '{0:.{1}f}'.format(prediction[0][1], 2)

        if float(output) > 0.5:
            return render_template(
                'forest.html',
                pred=f'🔥 Your Forest is in Danger.\nProbability of fire occurring is {output}'
            )
        else:
            return render_template(
                'forest.html',
                pred=f'🌲 Your Forest is Safe.\nProbability of fire occurring is {output}'
            )
    except Exception as e:
        return render_template('forest.html', pred=f"❌ Error occurred: {str(e)}")


@app.route('/precautions')
def precautions():
    return render_template("precautions.html")


@app.route('/info')
def info():
    return render_template("info.html")

@app.route('/map')
def map_page():
    return render_template("map.html")


if __name__ == '__main__':
    app.run(debug=True)
