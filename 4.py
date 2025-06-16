import pandas as pd
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Create synthetic dataset
positive_reviews = [
    "Amazing movie", "Loved the acting", "Great storyline", "Wonderful direction",
    "Beautiful cinematography", "Fantastic performance", "Highly entertaining",
    "Very enjoyable", "Best movie ever", "Incredible experience"
] * 5  # 50 positive

negative_reviews = [
    "Terrible movie", "Bad acting", "Poor storyline", "Awful direction",
    "Horrible cinematography", "Worst performance", "Not entertaining",
    "Very boring", "Worst movie ever", "Disappointing experience"
] * 5  # 50 negative

reviews = positive_reviews + negative_reviews
sentiments = ['positive'] * 50 + ['negative'] * 50

# Create DataFrame
df = pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 2: Vectorize with CountVectorizer
vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)

# Step 5: Define prediction function
def predict_review_sentiment(model, vectorizer, review):
    """
    Predict sentiment of a single review using trained model and fitted vectorizer.
    """
    vec = vectorizer.transform([review])
    prediction = model.predict(vec)[0]
    return prediction

# Example usage
example_review = "The plot was thrilling and well-acted"
predicted_sentiment = predict_review_sentiment(mode_
