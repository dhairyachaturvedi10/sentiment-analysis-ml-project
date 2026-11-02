import joblib
import re
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
nltk.download('stopwords', quiet=True)

# 1. Load the saved model and vectorizer
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
MODEL_PATH = "models/naive_bayes_model.pkl"

try:
    print("Loading models...")
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    print("Models loaded successfully!\n")
except FileNotFoundError:
    print("Error: Model files not found. Please run train.py first.")
    exit()

# 2. Recreate the exact cleaning function used in preprocessing
stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.discard('no')

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# 3. Prediction Function
def predict_sentiment(review_text):
    # Clean the input text
    cleaned_review = clean_text(review_text)
    
    # Vectorize the cleaned text using the loaded TF-IDF vectorizer
    # Note: We use .transform(), NOT .fit_transform()
    vectorized_text = vectorizer.transform([cleaned_review])
    
    # Make prediction (0 = Negative, 1 = Positive)
    prediction = model.predict(vectorized_text)[0]
    
    # Get prediction probability (confidence score)
    probabilities = model.predict_proba(vectorized_text)[0]
    confidence = probabilities[prediction] * 100
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment, confidence

# 4. Interactive testing loop
if __name__ == "__main__":
    print("-" * 50)
    print("Sentiment Analysis Predictor (Type 'exit' to quit)")
    print("-" * 50)
    
    while True:
        user_input = input("\nEnter a product review: ")
        
        if user_input.lower() == 'exit':
            print("Exiting predictor. Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        sentiment, confidence = predict_sentiment(user_input)
        
        print(f"Prediction: {sentiment}")
        print(f"Confidence: {confidence:.2f}%")