from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news"]
        transformed = vectorizer.transform([news_text])
        prediction = model.predict(transformed)[0]

        if prediction == "REAL":
            prediction_class = "success"
        else:
            prediction_class = "danger"

        return render_template(
            "index.html",
            prediction_text=f"This news is: {prediction}",
            prediction_class=prediction_class
        )

if __name__ == "__main__":
    app.run(debug=True)
