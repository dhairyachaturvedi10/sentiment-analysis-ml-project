from flask import Flask, request, jsonify, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load trained models
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/naive_bayes_model.pkl")

# NLP setup
stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.discard('no')

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data.get('review', '')
    
    if not review.strip():
        return jsonify({'error': 'No text provided'})
        
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0][prediction] * 100
    
    return jsonify({
        'sentiment': 'Positive' if prediction == 1 else 'Negative',
        'confidence': round(confidence, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)