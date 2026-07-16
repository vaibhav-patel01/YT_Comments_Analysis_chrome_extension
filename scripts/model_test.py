import mlflow.pyfunc
import pytest
import os
import dagshub
from mlflow.tracking import MlflowClient

# Set your remote tracking URI
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

@pytest.mark.parametrize("model_name, stage", [
    ("YT_comments_chrome_plugin", "staging"),])
def test_load_latest_staging_model(model_name, stage):
    client = MlflowClient()

    # Get the latest version in the specified stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in the '{stage}' stage for '{model_name}'"

    try:
        # Load the latest version of the model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Ensure the model loads successfully
        assert model is not None, "Model failed to load"
        print(f"Model '{model_name}' version {latest_version} loaded successfully from '{stage}' stage.")
        # Point the 'production' alias to Version 3
        client.set_registered_model_alias(
            name=model_name, 
            alias="production", 
            version=latest_version
        )
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")