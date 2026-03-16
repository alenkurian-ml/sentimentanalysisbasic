"""
STEP 1: Train a simple sentiment analyzer
This learns from examples to understand positive vs negative reviews
"""

# Import libraries (install with: pip install scikit-learn)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Our training data - just 10 examples!
# Format: (review text, label)
# 1 = Positive, 0 = Negative

reviews = [
    "This movie was great I loved it",
    "Awesome film really enjoyed",
    "Best movie ever amazing",
    "Wonderful story and acting",
    "Fantastic must watch",
    "Terrible movie hated it",
    "Worst film ever seen",
    "Boring and bad acting",
    "Awful waste of time",
    "Horrible do not watch"
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=positive, 0=negative

print("Training sentiment analyzer...")
print(f"Learning from {len(reviews)} examples\n")

# STEP 1: Convert text to numbers
# Computer can't understand words, only numbers!
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

print("Converted words to numbers")

# STEP 2: Train the model
# This is where the "learning" happens!
model = MultinomialNB()
model.fit(X, labels)

print("Model trained!")

# STEP 3: Save the model so we can use it later
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved!\n")
print("=" * 50)
print("SUCCESS! Now run predict.py to test it")
print("=" * 50)