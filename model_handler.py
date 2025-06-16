"""
Model loading and prediction module.
"""
import logging
import joblib
from config import MODEL_PATH

def load_model():
    """
    Load the pre-trained LightGBM model from disk.
    Returns:
        Loaded model object
    """
    try:
        model = joblib.load(MODEL_PATH)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def predict(model, X):
    """
    Make predictions using the loaded model.
    Args:
        model: Trained model
        X: Features DataFrame
    Returns:
        Predictions array
    """
    try:
        preds = model.predict(X)
        logging.info("Predictions made successfully.")
        return preds
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise
