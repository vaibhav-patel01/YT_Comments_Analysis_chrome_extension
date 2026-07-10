import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml

# extracting the parameters
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)
test_size = params["data_ingestion"]["test_size"]

# loading data
df = pd.read_csv("hf://datasets/AmaanP314/youtube-comment-sentiment/youtube-comments-sentiment.csv",
                usecols=["CommentText", "Sentiment"])

# removing duplicates
df.drop_duplicates(inplace= True)

df.drop_duplicates(subset=["CommentText"] , inplace = True)
# remove null values if any
df.dropna(inplace= True)

# mapping into numbers for lightgbm model
df["Sentiment"] = df["Sentiment"].map({"Neutral" : 1, "Positive" : 2, "Negative" : 0})
# remove leading and trailing spaces
df["CommentText"] = df["CommentText"].str.strip()

# split test and train
train_data, test_data = train_test_split(df, test_size= test_size, random_state=42)

# save files in data folder
data_path = os.path.join("data", "raw")

os.makedirs(data_path, exist_ok=True)

train_data.to_csv(os.path.join(data_path, "train_data.csv"), index = False)
test_data.to_csv(os.path.join(data_path, "test_data.csv"), index = False)