"""
Responsible AI dashboard setup module.
"""
import logging
import pandas as pd
import numpy as np
from raiwidgets import ResponsibleAIDashboard, FairnessDashboard
from responsibleai import RAIInsights

def launch_rai_dashboard(model, csv_path):
    """
    Initialize and launch RAI dashboards with causal, counterfactual, and fairness components.
    
    Args:
        model: Trained model
        csv_path: Path to the CSV file containing features and predictions
    """
    try:
        # Load data from CSV
        df = pd.read_csv(csv_path)

        # Ensure target column is 1D
        target_col = "predicted_future_collision"
        df[target_col] = np.ravel(df[target_col].values)

        # Limit to top 1000 for performance
        df = df.head(1000)

        # Prepare target
        y_true = df[target_col]
        y_pred = df[target_col]  # same as predicted in prediction-only mode

        # Try RAIInsights init with y_test (fallback if not supported)
        try:
            rai_insights = RAIInsights(
                model=model,
                train=df,
                test=df,
                target_column=target_col,
                task_type='classification',
                y_test=np.ravel(y_true.values)
            )
        except TypeError:
            rai_insights = RAIInsights(
                model=model,
                train=df,
                test=df,
                target_column=target_col,
                task_type='classification'
            )

        # Suppress LightGBM warnings
        try:
            model.set_params(verbosity=-1)
        except Exception:
            pass  # model may not support set_params

        # Add causal & counterfactual modules
        rai_insights.causal.add(treatment_features=[
            "hard_acceleration",
            "very_hard_acceleration",
            "extreme_acceleration",
            "hard_braking",
            "very_hard_braking",
            "extreme_braking",
            "hard_acceleration_while_cornering",
            "hard_braking_while_cornering"
        ])
        rai_insights.counterfactual.add(total_CFs=10, desired_class="opposite")

        # Add error analysis module
        rai_insights.error_analysis.add()

        # Compute RAI modules
        rai_insights.compute()
        logging.info("RAIInsights computed.")
        # Launch ResponsibleAI Dashboard
        ResponsibleAIDashboard(rai_insights)

    except Exception as e:
        logging.error(f"Error launching RAI dashboard: {e}")
        raise

def launch_fairness_dashboard(model, csv_path):
    """
    Initialize and launch Fairness dashboard.
    
    Args:
        model: Trained model
        csv_path: Path to the CSV file containing features and predictions
    """
    try:
        # Load data from CSV
        df = pd.read_csv(csv_path)

        # Ensure target column is 1D
        target_col = "predicted_future_collision"
        df[target_col] = np.ravel(df[target_col].values)

        # Limit to top 1000 for performance
        df = df.head(1000)

        # Prepare target
        y_true = df[target_col]
        y_pred = df[target_col]  # same as predicted in prediction-only mode

        # Detect sensitive features for FairnessDashboard
        sensitive_cols = [
            "white_label",
            "cluster",
            "city",
            "StartingLocation",
            "most_travelled_across_state",
            "IsCameraInstalled"
        ]
        sensitive_cols = [col for col in sensitive_cols if col in df.columns]
        
        if sensitive_cols:
            sensitive_features = df[sensitive_cols]
            FairnessDashboard(
                sensitive_features=sensitive_features,
                y_true=y_true,
                y_pred=y_pred
            )
            logging.info(f"FairnessDashboard launched using sensitive features: {sensitive_cols}")
        else:
            logging.warning("FairnessDashboard skipped — no sensitive features found.")

        input("Dashboards are running. Press Enter to exit...")
    except Exception as e:
        logging.error(f"Error launching Fairness dashboard: {e}")
        raise
# This module provides functions to launch the Responsible AI and Fairness dashboards.