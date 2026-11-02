import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_sentiment_model(data_path, vectorizer_save_path, model_save_path):
    print("Loading processed data...")
    df = pd.read_csv(data_path)
    
    # Drop any rows where Cleaned_Text became NaN after cleaning (e.g., reviews with only emojis/numbers)
    df.dropna(subset=['Cleaned_Text'], inplace=True)
    
    X = df['Cleaned_Text']
    y = df['Sentiment']
    
    print("Splitting data into 80% Train and 20% Test...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing text using TF-IDF...")
    # max_features=10000 keeps the top 10,000 most frequent words, preventing memory bloat and noise
    vectorizer = TfidfVectorizer(max_features=10000) 
    
    # Fit on training data and transform it, then only transform the test data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    
    print("Evaluating the model...")
    predictions = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=["Negative", "Positive"]))
    
    print("Saving vectorizer and model for future predictions...")
    # Ensure the models directory exists
    os.makedirs(os.path.dirname(vectorizer_save_path), exist_ok=True)
    joblib.dump(vectorizer, vectorizer_save_path)
    joblib.dump(model, model_save_path)
    print("Training complete and models saved successfully!")

if __name__ == "__main__":
    PROCESSED_DATA = "data/processed/cleaned_reviews.csv"
    VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
    MODEL_PATH = "models/naive_bayes_model.pkl"
    
    train_sentiment_model(PROCESSED_DATA, VECTORIZER_PATH, MODEL_PATH)