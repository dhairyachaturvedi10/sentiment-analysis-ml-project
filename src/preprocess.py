import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords list (only needs to be run once)
nltk.download('stopwords')

def prepare_and_clean_data(input_path, output_path, sample_size=10000):
    print("Loading raw dataset...")
    # Load dataset (using a subset of columns to save memory)
    df = pd.read_csv(input_path, usecols=['Score', 'Text'])
    
    # 1. Map Scores to Binary Sentiment
    print("Filtering neutral reviews and mapping sentiments...")
    # Drop neutral 3-star reviews
    df = df[df['Score'] != 3].copy()
    
    # Map 4 and 5 stars to Positive (1), 1 and 2 stars to Negative (0)
    df['Sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
    
    # Drop missing values
    df.dropna(subset=['Text'], inplace=True)
    
    # Sample the dataset for the mini-project to ensure fast training
    # Balance the dataset (50% Positive, 50% Negative)
    pos_reviews = df[df['Sentiment'] == 1]
    neg_reviews = df[df['Sentiment'] == 0]
    
    # Calculate samples per class (half of total, limited by the smaller class)
    samples_per_class = min(sample_size // 2, len(neg_reviews), len(pos_reviews))
    
    # Concatenate an equal amount of both, then shuffle the rows
    df = pd.concat([
        pos_reviews.sample(n=samples_per_class, random_state=42),
        neg_reviews.sample(n=samples_per_class, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Text Cleaning Pipeline
    print("Initializing NLP cleaning pipeline...")
    # Load English stopwords (words that carry no sentiment like 'is', 'the')
    stop_words = set(stopwords.words('english'))
    # Keep negation words because they flip sentiment (e.g., "not good")
    stop_words.discard('not')
    stop_words.discard('no')

    def clean_text(text):
        # Remove HTML tags (e.g., <br />)
        text = re.sub(r'<.*?>', ' ', text)
        # Remove anything that is not a letter (numbers, punctuation)
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove stopwords and reconstruct the sentence
        words = text.split()
        words = [word for word in words if word not in stop_words]
        return ' '.join(words)
    
    print("Applying text cleaning (this may take a minute)...")
    df['Cleaned_Text'] = df['Text'].apply(clean_text)
    
    # 3. Save Processed Data
    print(f"Saving processed data to {output_path}...")
    # Save only the necessary columns for training
    df[['Cleaned_Text', 'Sentiment']].to_csv(output_path, index=False)
    print("Data cleaning complete!")

# Execution block
if __name__ == "__main__":
    # Adjust these paths to match your file structure
    RAW_DATA_PATH = "data/raw/Reviews.csv/Reviews.csv"
    PROCESSED_DATA_PATH = "data/processed/cleaned_reviews.csv"
    
    prepare_and_clean_data(RAW_DATA_PATH, PROCESSED_DATA_PATH, sample_size=100000)