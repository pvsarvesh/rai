# Lightweight Production Pipeline for Responsible AI

## Overview
This pipeline connects to an Amazon RDS MySQL database, fetches and preprocesses data, loads a pre-trained LightGBM model, makes predictions, and launches Responsible AI dashboards (including fairness and causal analysis). It is optimized for production and can run efficiently on most instances.

## Quick Start
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure your database and model settings:**
   - Edit `config.py` with your DB credentials, table name, and model path.
3. **Run the pipeline:**
   ```bash
   python main.py
   ```

## Files
- `main.py`: Orchestrates the pipeline.
- `db_connector.py`: Handles DB connection and data fetching.
- `preprocessing.py`: Cleans and preprocesses the data.
- `model_handler.py`: Loads the model.
- `rai_dashboard.py`: Launches RAI dashboards.
- `config.py`: Stores configuration constants.
- `requirements.txt`: Minimal dependencies.

## Notes
- Only the features used by the model are used for prediction; sensitive features are retained for fairness analysis.
- The dashboard data is limited to 1000 rows for performance.
- Logging is concise and production-ready.

## Customization
- Update `TRAINED_FEATURES` and `DROP_COLS` in `config.py` as needed.
- Add or remove sensitive features in `rai_dashboard.py` as appropriate for your context.
