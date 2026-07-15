import json
import dagshub
import mlflow
import os

# dagshub.init(repo_owner='vaibhav-patel01', repo_name='YT_Comments_Analysis_chrome_extension', mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/vaibhav-patel01/YT_Comments_Analysis_chrome_extension.mlflow")

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = 'vaibhav-patel01'
repo_name = 'YT_Comments_Analysis_chrome_extension'

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

with open("experiment_info.json", 'r') as file:
    model_info = json.load(file)

model_name = "YT_comments_chrome_plugin"

# Consume the exact URI handed over by the evaluation script
model_uri = model_info['model_uri']

# Register the model
model_version = mlflow.register_model(model_uri, model_name)
        
# Transition the model to "Staging" stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging"
)

with open("registration_status.log", "w") as log_file:
    log_file.write(f"Successfully registered {model_name} version {model_version.version} to Staging.")