"""
app.py — Flask backend for Bank Fraud Detection UI
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, request, jsonify
import joblib
import json
import numpy as np

app = Flask(__name__)

# ── Load saved assets ──────────────────────────────────────
rf_model = joblib.load("model_rf.pkl")
scaler   = joblib.load("scaler.pkl")

with open("feature_columns.json") as f:
    feature_columns = json.load(f)

with open("label_mappings.json") as f:
    label_mappings = json.load(f)

# ── Split categoricals: dropdown (<=50 values) vs free-text ──
DROPDOWN_MAX  = 50
dropdown_cols = {k: v for k, v in label_mappings.items() if len(v) <= DROPDOWN_MAX}
freetext_cats = {k: v for k, v in label_mappings.items() if len(v) >  DROPDOWN_MAX}

print("✅ All models and assets loaded.")
print(f"   Features        : {feature_columns}")
print(f"   Dropdowns       : {list(dropdown_cols.keys())}")
print(f"   Free-text cats  : {list(freetext_cats.keys())}")


# ── /options — tells the UI what fields and choices to render ──
@app.route("/options", methods=["GET"])
def get_options():
    numeric_cols = [c for c in feature_columns if c not in label_mappings]
    return jsonify({
        "categorical":   dropdown_cols,
        "freetext_cats": list(freetext_cats.keys()),
        "numeric":       numeric_cols,
        "features":      feature_columns
    })


# ── /predict ───────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_vec = []

        for col in feature_columns:
            val = data.get(col)
            if val is None or str(val).strip() == "":
                return jsonify({"error": f"Missing field: {col}"}), 400

            if col in dropdown_cols:
                classes = label_mappings[col]
                if val not in classes:
                    return jsonify({"error": f"Invalid value '{val}' for '{col}'. Expected: {classes}"}), 400
                input_vec.append(classes.index(val))

            elif col in freetext_cats:
                classes = label_mappings[col]
                idx = classes.index(val) if val in classes else 0
                input_vec.append(idx)

            else:
                try:
                    input_vec.append(float(val))
                except ValueError:
                    return jsonify({"error": f"'{col}' must be a number, got '{val}'"}), 400

        arr    = np.array(input_vec).reshape(1, -1)
        scaled = scaler.transform(arr)

        prediction  = int(rf_model.predict(scaled)[0])
        probability = float(rf_model.predict_proba(scaled)[0][1])

        return jsonify({
            "prediction":  prediction,
            "probability": round(probability * 100, 2),
            "label":       "FRAUD" if prediction == 1 else "LEGITIMATE",
            "confidence":  round(max(probability, 1 - probability) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── serve the UI ───────────────────────────────────────────
@app.route("/")
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app.run(debug=True, port=5000)