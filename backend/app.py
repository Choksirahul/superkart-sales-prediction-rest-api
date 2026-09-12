import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Prediction API")

# Load the complete preprocessing + model pipeline
model = joblib.load("superkart_sales_prediction_model_v1_0.joblib")

EXPECTED_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category"
]

@superkart_api.get("/")
def home():
    return "Welcome to the SuperKart Sales Prediction API!"


@superkart_api.post("/v1/predict")
def predict_sales():
    try:
        product_data = request.get_json()

        # Build a one-row DataFrame in the same feature order used during training
        input_data = pd.DataFrame(
            [{feature: product_data[feature] for feature in EXPECTED_FEATURES}]
        )

        predicted_sales = round(float(model.predict(input_data)[0]), 2)

        return jsonify({
            "Predicted Product Store Sales Total": predicted_sales
        })

    except KeyError as error:
        return jsonify({
            "error": f"Missing required feature: {error.args[0]}"
        }), 400

    except Exception as error:
        return jsonify({"error": str(error)}), 400


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    try:
        if "file" not in request.files:
            return jsonify({"error": "CSV file is required."}), 400

        file = request.files["file"]
        input_data = pd.read_csv(file)

        missing_columns = [
            col for col in EXPECTED_FEATURES
            if col not in input_data.columns
        ]

        if missing_columns:
            return jsonify({
                "error": f"Missing required columns: {missing_columns}"
            }), 400

        input_data = input_data[EXPECTED_FEATURES]
        predictions = model.predict(input_data)

        output_dict = {
            str(index): round(float(prediction), 2)
            for index, prediction in enumerate(predictions)
        }

        return jsonify(output_dict)

    except Exception as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=True)
