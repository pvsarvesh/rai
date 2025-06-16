"""
Main pipeline orchestration script.
"""
import logging
import tempfile
import os
import pandas as pd
import numpy as np
from db_connector import get_db_engine, fetch_table_data
from preprocessing import preprocess_data
from model_handler import load_model, predict
from rai_dashboard import launch_rai_dashboard, launch_fairness_dashboard
from config import TABLE_NAME, TRAINED_FEATURES

def setup_logging():
    """
    Set up logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )

def main():
    setup_logging()
    logging.info("Pipeline started.")
    try:
        # Connect to DB and fetch data
        engine = get_db_engine()
        df = fetch_table_data(engine, TABLE_NAME)

        # Preprocess data
        X, y = preprocess_data(df)
        if y is not None:
            y = np.ravel(y)

        # Load model
        model = load_model()
        model.set_params(verbosity=-1)  # Suppress LightGBM console warnings

        # Filter and validate features
        missing_cols = [col for col in TRAINED_FEATURES if col not in X.columns]
        if missing_cols:
            raise ValueError(f"These required features are missing: {missing_cols}")

        X = X[TRAINED_FEATURES]

        logging.info(f"Features used for prediction: {TRAINED_FEATURES}")
        logging.info(f"Data shape after preprocessing: {X.shape}")

        # Make predictions
        predictions = model.predict(X)
        predictions = np.ravel(predictions)
        logging.info(f"Made predictions on {len(X)} records.")
        print(f"Predictions on {len(X)} records:\n{predictions[:10]}")

        # Save predictions to a temporary file
        temp_df = X.copy()
        temp_df["predicted_future_collision"] = predictions

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8") as tmp_file:
            temp_csv_path = tmp_file.name
            temp_df.to_csv(temp_csv_path, index=False)
            logging.info(f"Temporary prediction file saved at: {temp_csv_path}")

        # Launch RAI dashboard using the temp file
        launch_rai_dashboard(model, temp_csv_path)
        
        sens_df = df.copy()
        sens_df["predicted_future_collision"] = predictions

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", newline="", encoding="utf-8") as tmp_file:
            sens_csv_path = tmp_file.name
            sens_df.to_csv(sens_csv_path, index=False)
            logging.info(f"Temporary sensitive file saved at: {sens_csv_path}")
        
        # Launch Fairness dashboard
        launch_fairness_dashboard(model, sens_csv_path)

        # Delete temp file after dashboard closes
        try:
            os.remove(temp_csv_path)
            logging.info("Temporary file deleted.")
            os.remove(sens_csv_path)
            logging.info("Temporary sensitive file deleted.")
        except Exception as e:
            logging.warning(f"Could not delete temp file: {e}")

        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == "__main__":
    main()