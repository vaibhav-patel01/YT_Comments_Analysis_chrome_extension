from fastapi import FastAPI , HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import regex as re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import mlflow
import dagshub
import os

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

model_name = "YT_comments_chrome_plugin"
model_uri = f"models:/{model_name}@production"
model = mlflow.sklearn.load_model(model_uri)

# load the vectorizer

with open("tfidf_vectorizer.pkl", 'rb') as file : 
    vectorizer = pickle.load(file)


# data preprocessing

stopword = set(stopwords.words("english")) - {"not", "no", "but","against","because", "too", "more", "very", "most", "few","don", "don't", "ain", "aren", "aren't", "couldn", "couldn't", 
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", 
    "hasn't", "haven", "haven't", "isn", "isn't", "mightn", "mightn't", 
    "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", 
    "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", 
    "wouldn", "wouldn't" }

lemmetizer = WordNetLemmatizer()

def preprocessing(comment : str) : 
    comment = comment.lower().strip()
    
    # Use re.sub for regex replacements on standard strings
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    comment = re.sub(url_pattern, " ", comment)
    comment = re.sub(r'[\s\xa0]+', ' ', comment)
    comment = re.sub(r'[^a-zA-Z\...\!\~\s\?]', '', comment)
    comment = re.sub(r'\s+', ' ', comment).strip()
    
    # Process stopwords and lemmatize using standard list comprehensions, NOT .apply()
    words = comment.split()
    words = [word for word in words if word not in stopword]
    words = [lemmetizer.lemmatize(word, pos='v') for word in words]
    
    return ' '.join(words)


class CommentRequest(BaseModel):
    comments: List[str]

    
app = FastAPI(title= "YT_Comments_sentiment_analysis")

# aading middleware to talk with chrome extension

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def home():
    return {
        "status": "success",
        "message": "API is up and running."
    }


@app.post("/predict")
def predict_sentiment(data : CommentRequest):
    if not data.comments:
        raise HTTPException(status_code=400, detail="No comments.")
    
    try:
        cleaned_comments = [preprocessing(c) for c in data.comments]
        # 1. Transform the incoming text comments into numerical vectors
        vectorized_data = vectorizer.transform(cleaned_comments)
        
        # 2. Run the vectorized data through the MLflow model
        predictions = model.predict(vectorized_data)
        
        # 3. Zip the original comments with their predicted labels for the JSON response
        results = []
        for comment, pred in zip(data.comments, predictions):
            results.append({
                "comment": comment,
                "sentiment": int(pred)  # Cast to standard int (NumPy types can sometimes break JSON serialization)
            })
            
        return JSONResponse(status_code= 200, content={"results": results})
        
    except Exception as e:
        # Catch and return any processing errors (e.g., shape mismatches)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

