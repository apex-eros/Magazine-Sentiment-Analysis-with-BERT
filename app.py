from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertModel
import keras
from keras.preprocessing.sequence import pad_sequences

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = TFDistilBertModel.from_pretrained("distilbert-base-uncased")
loaded_model = keras.models.load_model("Best_sentiment_model.h5")

# Define a function to process the input review and make a prediction
def predict_review(review):
    # Tokenize and encode the review text
    inputs = tokenizer([review], padding=True, truncation=True, max_length=64, return_tensors="tf")
    outputs = bert_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    # Use the [CLS] token's representation for the classification
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    
    # Make the prediction
    prediction = np.argmax(loaded_model.predict(embedding), axis=1)[0]
    
    # Map predictions to the corresponding labels
    labels = {
        0: "Worst. The reader didn't like it at all.",
        1: "Bad. The reader didn't enjoy it.",
        2: "Neutral. Didn't feel that great.",
        3: "Good. Liked it.",
        4: "Loved it."
    }
    
    return labels.get(prediction, "Error")

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form['review']
        sentiment = predict_review(review_text)  # Call the prediction function
        return render_template('index.html', sentiment=sentiment)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
