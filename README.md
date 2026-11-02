# AI Sentiment Analysis Classifier

## Problem Statement
Manually analyzing customer feedback is unscalable. This Machine Learning pipeline categorizes unstructured e-commerce product reviews into "Positive" or "Negative" sentiments using Natural Language Processing (NLP), enabling automated tracking of customer satisfaction.

## System Architecture
* **Algorithm:** Multinomial Naive Bayes
* **Feature Extraction:** TF-IDF Vectorization (10,000 max features)
* **Preprocessing:** Regex HTML stripping & NLTK Stopwords removal
* **Dataset:** Amazon Fine Food Reviews
* **Web Framework:** Flask + Gunicorn

## Project Structure
* `/models`: Contains the serialized `.pkl` machine learning models.
* `/src`: Contains the data preprocessing, model training, and visualization scripts.
* `/report/images`: Contains the Confusion Matrix and Word Clouds.
* `app.py`: The main Flask backend server.

## Local Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Access the web interface at `http://127.0.0.1:5000`