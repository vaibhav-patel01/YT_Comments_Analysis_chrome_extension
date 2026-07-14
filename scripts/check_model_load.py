import mlflow
import dagshub

dagshub.init(
    repo_owner= 'vaibhav-patel01' ,
    repo_name= 'YT_Comments_Analysis_chrome_extension' ,
    mlflow= True
)

mlflow.set_tracking_uri("https://dagshub.com/vaibhav-patel01/YT_Comments_Analysis_chrome_extension.mlflow")

model_name = "YT_comments_chrome_plugin"
model_version = '2'

model_uri = f"models:/{model_name}/{model_version}"

model = mlflow.sklearn.load_model(model_uri)

