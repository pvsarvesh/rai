import pickle

with open("lgbm_model.pkl", "rb") as f:
    lgbm_model = pickle.load(f)

# Check for underlying Booster and extract feature names
if hasattr(lgbm_model, "booster_"):
    features = lgbm_model.booster_.feature_name()
    print("Features used during training:")
    for f in features:
        print(f)
else:
    print("Model does not have 'booster_' attribute. Make sure it was trained and saved properly.")
