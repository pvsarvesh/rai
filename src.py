import pandas as pd
import pickle
import sys
from raiwidgets import ResponsibleAIDashboard, FairnessDashboard
from responsibleai import RAIInsights

def main():
    try:
        # Load model
        with open("lgbm_model.pkl", "rb") as f:
            lgbm_model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    try:
        # Load dataset
        te = pd.read_csv("testing.csv")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    # Define target and columns to drop
    target_col = 'future_collision'
    drop_cols = [
        'whitelabel', 'DriverID', 'date', 'future_collision', 'collision',
        'low_acceleration', 'low_braking',
        'low_acceleration_while_cornering', 'low_braking_while_cornering',
        'average_speed'
    ]

    # Select feature columns
    feature_cols = [col for col in te.columns if col not in drop_cols]

    # Round feature columns
    for col in feature_cols:
        te[col] = te[col].apply(lambda x: round(x, 3))

    # Prepare test data
    test_data = te[feature_cols].copy()
    test_data[target_col] = te[target_col]

    # Create RAIInsights object
    try:
        rai_insights = RAIInsights(
            model=lgbm_model,
            train=test_data,
            test=test_data,
            target_column=target_col,
            task_type='classification'
        )
        rai_insights.causal.add(treatment_features=["hard_acceleration", "hard_braking"])
        rai_insights.counterfactual.add(total_CFs=10, desired_class="opposite")
        rai_insights.compute()
    except Exception as e:
        print(f"Error creating or computing RAIInsights: {e}")
        sys.exit(1)

    # Launch dashboards
    try:
        print("Launching FairnessDashboard...")
        FairnessDashboard(
            sensitive_features=te[['whitelabel']],
            y_true=te[target_col],
            y_pred=lgbm_model.predict(te[feature_cols])
        )
        
        print("Launching ResponsibleAIDashboard...")
        ResponsibleAIDashboard(rai_insights)

    except Exception as e:
        print(f"Error launching dashboards: {e}")
        sys.exit(1)

    # Keep the script running so dashboards remain accessible
    input("Dashboards are running. Press Enter to stop and close the dashboards...")

if __name__ == "__main__":
    main()