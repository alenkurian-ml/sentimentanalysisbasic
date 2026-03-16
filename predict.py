"""
STEP 2: Use the trained model to predict sentiment
"""

import pickle

# Load the saved model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model loaded!\n")

# Function to predict sentiment
def predict_sentiment(text):
    # Convert text to numbers (same way we trained)
    text_vector = vectorizer.transform([text])
    
    # Predict: 1 = positive, 0 = negative
    prediction = model.predict(text_vector)[0]
    
    # Get confidence (probability)
    confidence = model.predict_proba(text_vector)[0]
    
    if prediction == 1:
        sentiment = "POSITIVE"
        score = confidence[1] * 100
    else:
        sentiment = "NEGATIVE"
        score = confidence[0] * 100
    
    return sentiment, score

# Test with some examples
print("=" * 50)
print("SENTIMENT ANALYZER")
print("=" * 50)

# Test reviews
test_reviews = [
    "This movie is amazing",
    "I hated this film",
    "Great story and visuals",
    "Waste of money terrible"
]

for review in test_reviews:
    sentiment, confidence = predict_sentiment(review)
    print(f"\nReview: {review}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.1f}%")

# Interactive mode - type your own review!
print("\n" + "=" * 50)
print("Try it yourself!")
print("=" * 50)

while True:
    user_input = input("\nEnter a movie review (or 'quit' to exit): ")
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    if user_input.strip():
        sentiment, confidence = predict_sentiment(user_input)
        print(f"-> {sentiment} ({confidence:.1f}% confident)")