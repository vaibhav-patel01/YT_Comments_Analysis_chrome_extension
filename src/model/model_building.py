import os
import yaml
import pickle
import pandas as pd
from lightgbm import LGBMClassifier

processed_path = os.path.join("data","processed")
interim_path = os.path.join("data","interim")

# Load the features from the interim directory
with open(os.path.join(interim_path,"X_train.pkl"), "rb") as f:
    x_train = pickle.load(f)

# Load the processed CSV to extract the target variable
processed_train = pd.read_csv(os.path.join(processed_path,"processed_train_data.csv"))
y_train = processed_train["Sentiment"]

with open("params.yaml", 'r') as file :
    params = yaml.safe_load(file)

n_estimator = params["model_building"]["n_estimators"]
learning_rate = params["model_building"]["learning_rate"]
max_depth = params["model_building"]["max_depth"]
class_weight = params["model_building"]["class_weight"]
objective = params["model_building"]["objective"]
metric = params["model_building"]["metric"]

model = LGBMClassifier(
    n_estimators= n_estimator,
    learning_rate= learning_rate,
    max_depth= max_depth,
    class_weight = class_weight,
    objective= objective,
    metric = metric,
    verbose = -1
    )

model.fit(x_train, y_train)

model_path = os.path.join("model")

os.makedirs(model_path, exist_ok=True)

with open(os.path.join(model_path,'model.pkl'), 'wb') as f:
    pickle.dump(model, f)



