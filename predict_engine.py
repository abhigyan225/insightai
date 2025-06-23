# predict_engine.py

import pandas as pd
import pickle

def predict_from_uploaded_file(uploaded_file):
    try:
        # Load uploaded new data
        new_data = pd.read_csv(uploaded_file)

        # Load trained model
        with open("trained_model.pkl", "rb") as f:
            model = pickle.load(f)

        # Load expected input column structure
        with open("input_columns.pkl", "rb") as f:
            input_columns = pickle.load(f)

        # One-hot encode new data
        new_data = pd.get_dummies(new_data)

        # Add missing columns
        for col in input_columns:
            if col not in new_data.columns:
                new_data[col] = 0  # fill missing columns with zeros

        # Reorder columns to match training data
        new_data = new_data[input_columns]

        # Make predictions
        predictions = model.predict(new_data)

        # Add predictions to the DataFrame
        new_data["Prediction"] = predictions

        return new_data, "✅ Prediction successful!"

    except Exception as e:
        return None, f"❌ Prediction failed: {e}"
