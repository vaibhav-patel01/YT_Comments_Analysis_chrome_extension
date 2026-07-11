import os
import pandas as pd
import regex as re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# importing data
train_path = os.path.join("data", "raw", "train_data.csv")
test_path = os.path.join("data", "raw", "test_data.csv")

train_data = pd.read_csv(train_path)
test_data =pd.read_csv(test_path)

nltk.download("stopwords")
nltk.download("wordnet")

stopword = set(stopwords.words("english")) - {"not", "no", "but","against","because", "too", "more", "very", "most", "few","don", "don't", "ain", "aren", "aren't", "couldn", "couldn't", 
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", 
    "hasn't", "haven", "haven't", "isn", "isn't", "mightn", "mightn't", 
    "mustn", "mustn't", "needn", "needn't", "shan", "shan't", "shouldn", 
    "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", 
    "wouldn", "wouldn't" }

def clean_urls(df):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    df["CommentText"] = df["CommentText"].str.replace(url_pattern, " ", regex=True)
    df["CommentText"] = df["CommentText"].str.replace(r'[\s\xa0]+', ' ', regex=True)
    # strip again to be in safe side
    df["CommentText"] = df["CommentText"].str.strip()

    df["CommentText"] = df["CommentText"].str.lower()

    return df

def clean_text(df):
    if not isinstance(df, str):
            return ""
    
    # Keep ONLY: A-Z, a-z, spaces, and the literal characters . ! ~
    cleaned = re.sub(r'[^a-zA-Z\...\!\~\s\?]', '', df)
    
    # Collapse multiple spaces into one and strip the edges
    return re.sub(r'\s+', ' ', cleaned).strip()

def preprocessing(df):
    # removing all the urls and noises
    df = clean_urls(df)

    df["CommentText"] = df["CommentText"].apply(clean_text)

    # 2. Drop the rows that became empty, then reset the index
    df = df[df['CommentText'] != ""]
    df = df[df['CommentText'].str.strip() != ""]
    # 3. Reset index for a clean sequence
    df = df.reset_index(drop=True)

    # removing sstopwords
    # keeping the stopwords which may affect the sentiment
    df.dropna(inplace= True)

    df["CommentText"] = df["CommentText"].apply(
        lambda x : ' '.join([word for word in x.split() if word.lower() not in stopword])
    )


    # lemmetization
    # nltk.download("wordnet")
    lemmetizer = WordNetLemmatizer()
    df["CommentText"] = df["CommentText"].apply(lambda x : ' '.join([lemmetizer.lemmatize(word, pos='v') for word in x.split()]) )

    # making the data balanced 
    min_class_count = df['Sentiment'].value_counts().min()

    df = (df.groupby('Sentiment')
                .sample(n=min_class_count, random_state=42)
                .sample(frac=1, random_state=42) # Optional: Shuffle the final dataset
                .reset_index(drop=True))
    
    return df



# major preprocessing

processed_train_data = preprocessing(train_data)
processed_test_data = preprocessing(test_data)

# Convert any accidental NaNs to empty strings first so string methods
processed_train_data["CommentText"] = processed_train_data["CommentText"].fillna("")
processed_test_data["CommentText"] = processed_test_data["CommentText"].fillna("")

# Keep only rows where the text, when stripped of spaces, is NOT empty
processed_train_data = processed_train_data[processed_train_data["CommentText"].str.strip() != ""]
processed_test_data = processed_test_data[processed_test_data["CommentText"].str.strip() != ""]

# save files in data folder
data_path = os.path.join("data", "processed")

os.makedirs(data_path, exist_ok=True)

processed_train_data.to_csv(os.path.join(data_path, "processed_train_data.csv"), index = False)
processed_test_data.to_csv(os.path.join(data_path, "processed_test_data.csv"), index = False)



