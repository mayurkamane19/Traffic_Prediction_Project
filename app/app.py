from flask import Flask, render_template, request
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)

# Load trained model
model = pickle.load(open("../model/traffic_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    plot_file = None

    if request.method == "POST":
        # Read inputs
        values = [
            float(request.form["temp"]),
            float(request.form["rain"]),
            float(request.form["snow"]),
            float(request.form["clouds"]),
            int(request.form["hour"]),
            int(request.form["day"]),
            int(request.form["month"]),
            float(request.form["lag1"]),
            float(request.form["lag24"]),
        ]

        # Predict
        prediction = int(model.predict([values])[0])

        # ===============================
        # DYNAMIC MATPLOTLIB GRAPH
        # ===============================
        x = ["Traffic 1 hr ago", "Traffic 24 hrs ago", "Predicted"]
        y = [values[7], values[8], prediction]

        plt.figure(figsize=(6,4))
        plt.bar(x, y)
        plt.ylabel("Traffic Volume")
        plt.title("Traffic Prediction Comparison")

        # Unique filename
        plot_file = f"plot_{int(time.time())}.png"
        plot_path = os.path.join("static", plot_file)

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

    return render_template(
        "index.html",
        prediction=prediction,
        plot_file=plot_file
    )

if __name__ == "__main__":
    app.run(debug=True)
