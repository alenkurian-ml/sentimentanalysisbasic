"""
Gradio Web Interface (Compatible with Gradio 2.8.2 and Python 3.6)
Run with: python app.py
"""

import gradio as gr
import pickle

# Load model
try:
    with open('sentiment_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Error: Model files not found! Please run train.py first.")
    exit()

def predict_sentiment(text):
    """Predict sentiment of input text"""
    if not text.strip():
        return "Please enter some text!", ""
    
    # Transform and predict
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)[0]
    confidence = model.predict_proba(text_vector)[0]
    
    # Format result
    if prediction == 1:
        sentiment = "POSITIVE"
        conf_score = confidence[1] * 100
    else:
        sentiment = "NEGATIVE"
        conf_score = confidence[0] * 100
    
    result = f"**{sentiment}**"
    confidence_text = f"Confidence: {conf_score:.1f}%"
    
    return result, confidence_text

# Custom CSS for minimalist white and blue design
css = """
body, .gradio-container { background-color: white !important; }
.gr-button-primary { background-color: blue !important; color: white !important; border: none !important; }
.gr-button-secondary { background-color: white !important; color: blue !important; border: 1px solid blue !important; }
input, textarea { border: 1px solid blue !important; }
"""

# Create Gradio interface using the older Gradio 2.x API
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Type here...", label="Text Input"),
    outputs=[gr.outputs.Textbox(label="Sentiment"), gr.outputs.Textbox(label="Score")],
    css=css,
    allow_flagging="never"
)

# Launch the app
if __name__ == "__main__":
    interface.launch()