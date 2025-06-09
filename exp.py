## python 3.11.9
import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
import pickle


with open("lgbm_model.pkl", "rb") as f:
    lgbm_model = pickle.load(f)

te = pd.read_csv('testing.csv')

all_columns = te.columns.drop(['whitelabel','DriverID','date','future_collision','collision',
                              'low_acceleration','low_braking',
                               'low_acceleration_while_cornering','low_braking_while_cornering','average_speed'
                               ]).to_list()
for col in all_columns:
    te[col] = te[col].apply(lambda x:round(x,3))
explainer = ClassifierExplainer(lgbm_model, te[all_columns], te['future_collision'],
#                                 descriptions=feature_descriptions, # adds a table and hover labels to dashboard
                                labels=['No collision', 'Collision'], # defaults to ['0', '1', etc]
                                 index_name = "Driver behavior record no", # defaults to X.index.name
                                 target = "Collision", # defaults to y.name
                                )

db = ExplainerDashboard(explainer, 
                        title="Collision prevention Model Explainer", # defaults to "Model Explainer"
#                        shap_interaction=False # you can switch off tabs with bools
                        simple=True
                        )

#db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)



db.run(port=8050)