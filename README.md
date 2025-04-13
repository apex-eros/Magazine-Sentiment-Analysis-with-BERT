# Magazine-Sentiment-Analysis-with-BERT
This project performs sentiment analysis on magazine reviews using BERT embeddings and a neural network. It includes data preprocessing, model training, and evaluation. The model is deployed as a Flask web application, where users can input reviews and receive sentiment predictions.

This repository implements a sentiment analysis model for magazine reviews using BERT embeddings and a fully connected neural network (FFNN). The project includes:

Data Preprocessing: Loading and transforming review data into BERT-compatible embeddings.

Model Training: A neural network is trained to classify reviews into five sentiment categories: Worst, Bad, Neutral, Good, and Loved.

Flask Deployment: The model is deployed as a Flask web application where users can input reviews and get sentiment predictions in real-time.

Key Features:
Utilizes DistilBERT for extracting meaningful embeddings from reviews.

SMOTE technique used for handling class imbalance in the dataset.

Hyperparameter optimization using Keras Tuner for improved model performance.

Interactive web interface built with Flask for easy user interaction.

Getting Started:
Clone the repository.

Install the required dependencies. You can manually install the following:

tensorflow

transformers

keras

scikit-learn

imbalanced-learn

flask

keras-tuner

Add the index.html file in the templates folder.

Run the Flask app locally using python app.py
