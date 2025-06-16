import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from config import DROP_COLS, TARGET_COL

def preprocess_data(df):
    """
    Preprocess the input DataFrame: drop columns, round numerics, encode object columns (label encoding for high-cardinality), handle missing values, retain target if present.
    Returns:
        Tuple (X, y or None)
    """
    try:
        # Drop known irrelevant columns
        df = df.drop(columns=[col for col in DROP_COLS if col in df.columns], errors='ignore')
        logging.info(f"Dropped columns: {DROP_COLS}")

        # Handle missing values first (so encoding doesn't break)
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna('missing')

        # Encode object columns using LabelEncoder
        object_cols = df.select_dtypes(include='object').columns
        if len(object_cols) > 0:
            logging.info(f"Label encoding object columns: {list(object_cols)}")
            for col in object_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # Round numeric columns
        num_cols = df.select_dtypes(include=['float', 'float64', 'int', 'int64']).columns
        df[num_cols] = df[num_cols].round(2)
        logging.info("Rounded numeric columns.")

        # Rename columns to match model training
        RENAME_MAP = {
            "speedDiff_0": "change_in_velocity_0th_percentile",
            "speedDiff_1": "change_in_velocity_1st_percentile",
            "speedDiff_95": "change_in_velocity_95th_percentile",
            "speedDiff_99": "change_in_velocity_99th_percentile",
            "speed_99": "maximum_speed"
        }

        df.rename(columns=RENAME_MAP, inplace=True)


        # Separate features and target
        if TARGET_COL in df.columns:
            y = df[TARGET_COL].squeeze()
            y = np.ravel(y)
            X = df.drop(columns=[TARGET_COL])
            logging.info(f"Target column '{TARGET_COL}' found. Evaluation mode.")
            return X, y
        else:
            logging.warning(f"Target column '{TARGET_COL}' not found. Prediction-only mode.")
            return df, None

    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise