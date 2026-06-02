# utils/predict.py
import joblib
import pandas as pd
import numpy as np
import os

def load_artifacts(models_dir='models'):
    """
    Load the trained model, scaler, label encoder, and selected feature names.
    Returns (None, None, None, None) if any required file is missing or corrupted.
    """
    model_path = os.path.join(models_dir, 'best_model.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    encoder_path = os.path.join(models_dir, 'label_encoder.joblib')
    features_path = os.path.join(models_dir, 'feature_names.joblib')

    # Check if all files exist
    if not all(os.path.exists(p) for p in [model_path, scaler_path, encoder_path, features_path]):
        return None, None, None, None

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        features = joblib.load(features_path)
        return model, scaler, encoder, features
    except (EOFError, Exception) as e:
        # If any file is corrupted, return None and log the error
        print(f"⚠️ Error loading artifacts: {e}. Files may be corrupted.")
        return None, None, None, None


def predict_from_df(df, model, scaler, encoder, features):
    """
    Predict on a DataFrame that contains the required features.
    Returns a DataFrame with original data, prediction, confidence, and attack type.
    """
    if model is None or scaler is None:
        raise RuntimeError("Model or scaler not loaded. Please train the model first.")

    # Ensure we only use the selected features
    missing_features = set(features) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")

    X_input = df[features].copy()

    # Handle missing values (simple imputation like preprocessing)
    for col in X_input.columns:
        if X_input[col].dtype == 'object':
            X_input[col].fillna('Unknown', inplace=True)
        else:
            X_input[col].fillna(X_input[col].median(), inplace=True)

    # Convert all columns to numeric (if any categorical remain, encode them as numbers)
    X_input = X_input.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Scale
    X_scaled = scaler.transform(X_input)

    # Predict
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    confidence = np.max(probabilities, axis=1)

    # Decode numeric predictions back to original labels
    decoded = encoder.inverse_transform(predictions)

    # Create result DataFrame
    result_df = df.copy()
    result_df['Prediction'] = decoded
    result_df['Confidence'] = confidence

    # Determine if normal or attack (binary/multi: BENIGN means normal, otherwise attack)
    result_df['Status'] = result_df['Prediction'].apply(
        lambda x: 'Normal' if str(x).upper() == 'BENIGN' else 'Attack'
    )

    return result_df


def predict_manual(input_dict, model, scaler, encoder, features):
    """
    Predict for a single manual entry (dictionary of feature values).
    Returns the prediction label, confidence, and status.
    """
    if model is None or scaler is None:
        raise RuntimeError("Model or scaler not loaded. Please train the model first.")

    input_df = pd.DataFrame([input_dict])
    result = predict_from_df(input_df, model, scaler, encoder, features)
    pred_label = result['Prediction'].iloc[0]
    confidence = result['Confidence'].iloc[0]
    status = result['Status'].iloc[0]
    return pred_label, confidence, status