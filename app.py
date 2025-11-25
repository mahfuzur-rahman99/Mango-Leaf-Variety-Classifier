from flask import Flask, render_template, request, jsonify
from model import load_model, predict_image_bytes

app = Flask(__name__)

# Load model once at startup
model = load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        img_bytes = file.read()
        class_name, confidence = predict_image_bytes(model, img_bytes)

        return jsonify({
            "class_name": class_name,
            "confidence": round(confidence * 100, 2)  # percentage
        })
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
