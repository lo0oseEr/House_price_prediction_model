from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates')

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])
        output = round(prediction[0], 2)
        return render_template("index.html", prediction_text=f"The final price of the house is: Rs {output}")
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json(force=True)
        values = np.array(list(data.values())).reshape(1, -1)
        prediction = model.predict(values)
        output = round(prediction[0], 2)
        return jsonify({'predicted_price': output})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
