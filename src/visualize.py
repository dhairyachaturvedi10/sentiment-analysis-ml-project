import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib
import os

# Create a folder to save your report images
os.makedirs("report/images", exist_ok=True)

print("Loading data and models...")
df = pd.read_csv("data/processed/cleaned_reviews.csv")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/naive_bayes_model.pkl")

# 1. Generate Confusion Matrix
print("Generating Confusion Matrix...")
# Recreate the exact same test split used in training
X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Text'], df['Sentiment'], test_size=0.2, random_state=42)

# Predict on the test set
X_test_tfidf = vectorizer.transform(X_test)
predictions = model.predict(X_test_tfidf)

# Plot the matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Naive Bayes Confusion Matrix')
plt.ylabel('Actual Sentiment')
plt.xlabel('Predicted Sentiment')
plt.savefig("report/images/confusion_matrix.png", bbox_inches='tight')
plt.close()

# 2. Generate Word Clouds
print("Generating Word Clouds...")
positive_text = " ".join(df[df['Sentiment'] == 1]['Cleaned_Text'].astype(str))
negative_text = " ".join(df[df['Sentiment'] == 0]['Cleaned_Text'].astype(str))

# Positive Word Cloud
wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Positive Reviews", fontsize=16)
plt.savefig("report/images/positive_wordcloud.png", bbox_inches='tight')
plt.close()

# Negative Word Cloud
wordcloud_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words in Negative Reviews", fontsize=16)
plt.savefig("report/images/negative_wordcloud.png", bbox_inches='tight')
plt.close()

print("Success! Check the 'report/images' folder for your charts.")