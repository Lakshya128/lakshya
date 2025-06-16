import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score

# Step 1: Create a synthetic dataset of 100 short texts
good_texts = [
    "Great product", "Loved it", "Very useful", "Excellent quality",
    "Highly recommend", "Fantastic experience", "Good value", "Nice service",
    "Well made", "Fast delivery"
] * 5  # 50 good samples

bad_texts = [
    "Terrible product", "Hated it", "Not useful", "Bad quality",
    "Do not buy", "Awful experience", "Poor value", "Rude service",
    "Cheaply made", "Slow delivery"
] * 5  # 50 bad samples

texts = good_texts + bad_texts
labels = [1]*50 + [0]*50  # 1 for good, 0 for bad

# Shuffle the dataset
combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)

# Step 2: Preprocess the text using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=300, lowercase=True, stop_words='english')
X = vectorizer.fit_transform(texts)

# Step 3: Split into training and testing sets (75%/25%)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Step 4: Train Logistic Regression and evaluate
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Step 5: Write preprocessing function
def text_preprocess_vectorize(texts, vectorizer):
    """
    Takes a list of texts and a fitted TfidfVectorizer, and returns the vectorized matrix.
    """
    return vectorizer.transform(texts)

# Example usage
sample_texts = ["Amazing service", "Not worth the money"]
X_sample = text_preprocess_vectorize(sample_texts, vectorizer)
print("Vectorized sample shape:", X_sample.shape)