
import os
import yaml
import pickle
import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


warnings.filterwarnings("ignore", category=UserWarning)

path = os.path.join("data","processed")

train = pd.read_csv(os.path.join(path, "processed_train_data.csv"))
test = pd.read_csv(os.path.join(path, "processed_test_data.csv"))

X_train = train["CommentText"]
X_test = test["CommentText"]

with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)

ngram_range = tuple(params["feature_engineering"]["ngram_range"])

max_features = params["feature_engineering"]["max_features"]

vectorizer = TfidfVectorizer(ngram_range= ngram_range, max_features= max_features)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

save_path = os.path.join("data","interim")
os.makedirs(save_path, exist_ok=True)

# save the transformed data

with open(os.path.join(save_path, 'X_train.pkl'), 'wb') as f:
    pickle.dump(X_train, f)

with open(os.path.join(save_path, 'X_test.pkl'), 'wb') as f:
    pickle.dump(X_test, f)

# Save the fitted vectorizer object for future 
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print(f"Vectorizer and data saved successfully as .pkl files. Max features: {max_features}, N-grams: {ngram_range}")