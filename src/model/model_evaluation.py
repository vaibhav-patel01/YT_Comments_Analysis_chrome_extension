import os
import yaml
import pandas as pd
import json
import pickle
import dagshub
import mlflow
import mlflow.lightgbm
import plotly.express as px
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix 


# loading all the parameters

with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)

ngram_range = tuple(params["feature_engineering"]["ngram_range"])
max_features = params["feature_engineering"]["max_features"]
test_size = params["data_ingestion"]["test_size"]
n_estimator = params["model_building"]["n_estimators"]
learning_rate = params["model_building"]["learning_rate"]
max_depth = params["model_building"]["max_depth"]
class_weight = params["model_building"]["class_weight"]
objective = params["model_building"]["objective"]
metric = params["model_building"]["metric"]
features_tech = params["model_evaluation"]["features"]

# loading the data
with open("data/interim/X_test.pkl", 'rb') as file1 :
    X_test = pickle.load(file1)


y_test = pd.read_csv("data/processed/processed_test_data.csv")
y_test = y_test["Sentiment"]


# loading model
with open("model/model.pkl", 'rb') as file2:
    model = pickle.load(file2)

# setting up the remote mlflow
dagshub.init(repo_owner= 'vaibhav-patel01' , repo_name= 'YT_Comments_Analysis_chrome_extension' , mlflow= True)

mlflow.set_tracking_uri("https://dagshub.com/vaibhav-patel01/YT_Comments_Analysis_chrome_extension.mlflow")


mlflow.set_experiment("dvc_pipeline_runs")


with mlflow.start_run() :
        # log parameters
        mlflow.log_params({
             "test_size" : test_size,
             "features_algo" : features_tech,
             "n_grams" : ngram_range,
             "max_features" : max_features,
             "max_depth" : max_depth,
             "n_estimators" : n_estimator,
             "learning_rate" : learning_rate,
             "class_weight" : class_weight,
             "objective" : objective,
             "metric" : metric
        })
        # Make predictions and log metrics
        y_pred = model.predict(X_test)

        # Log accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log classification report
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in classification_rep.items():
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    mlflow.log_metric(f"{label}_{metric}", value)

        # Log confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        fig = px.imshow(
            conf_matrix,
            text_auto=True, # Equivalent to annot=True
            color_continuous_scale="Blues", # Equivalent to cmap="Blues"
            labels=dict(x="Predicted", y="Actual"),
            title= "Confusion Matrix:",
            width=800, # Approximate equivalent to figsize=(8, 6)
            height=600
        )
        mlflow.log_figure(fig, "basline_confusion_matrix.html")
        # logging model
        
        mlflow.lightgbm.log_model(
            lgb_model=model, 
            artifact_path="lgbm_model"
        )

print(accuracy)
print(classification_report(y_test, y_pred))