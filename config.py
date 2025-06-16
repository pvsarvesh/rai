"""
Configuration constants for the pipeline.
"""

# Database credentials
DB_HOST = 'cogo-insurance.cjuawrphqolo.us-east-1.rds.amazonaws.com'
DB_PORT = 3306
DB_USER = 'cogoinsurance'
DB_PASSWORD = '9k7ofPkde7qR'  
DB_NAME = 'cogoinsurance'

# Table name to fetch data from
TABLE_NAME = 'gpstab_driver_scores'

# Columns to drop during preprocessing
DROP_COLS = [
    'whitelabel', 'DriverID', 'date', 'future_collision', 'collision',
    'low_acceleration', 'low_braking',
    'low_acceleration_while_cornering', 'low_braking_while_cornering',
    'average_speed'
]

# Features that are trained on the model
TRAINED_FEATURES = [
    "hard_acceleration",
    "very_hard_acceleration",
    "extreme_acceleration",
    "hard_braking",
    "very_hard_braking",
    "extreme_braking",
    "hard_acceleration_while_cornering",
    "hard_braking_while_cornering",
    "change_in_velocity_0th_percentile",
    "change_in_velocity_1st_percentile",
    "change_in_velocity_95th_percentile",
    "change_in_velocity_99th_percentile",
    "maximum_speed",
    "high_speed_time",
    "excess_high_speed_time",
    "total_driving_hours",
    "total_night_hours"
]

# Target column for classification
TARGET_COL = 'future_collision'

# Path to the pre-trained LightGBM model
MODEL_PATH = 'lgbm_model.pkl'
