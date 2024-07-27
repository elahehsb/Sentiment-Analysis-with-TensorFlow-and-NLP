# 1. Import Libraries
# This section imports the necessary libraries for data manipulation, natural language processing, and deep learning. These include:

# numpy and pandas for numerical operations and data manipulation.
# tensorflow and keras for building and training the deep learning model.
# sklearn for splitting the data into training and testing sets.
# matplotlib and seaborn for data visualization.
# ------------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


# 2. Load and Explore the Dataset
# This section loads the IMDb movie reviews dataset and performs initial exploratory data analysis (EDA):

# Loading the Dataset: Load the dataset using pandas.
# Data Inspection: Print the first few rows and descriptive statistics to understand the data.
# Missing Values Check: Ensure there are no missing values in the dataset.
# ------------------------------------------------------------------

# Load the IMDb dataset
imdb_data = pd.read_csv('https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', compression='gzip', header=0, delimiter='\t', quoting=3)

# Display the first few rows of the dataset
print(imdb_data.head())

# Display dataset statistics
print(imdb_data.describe())

# Check for missing values
print(imdb_data.isnull().sum())


# 3. Data Preprocessing
# This section prepares the data for model training:

# Feature Selection: Separate features (X) and target variable (y).
# Train-Test Split: Split the data into training and testing sets to evaluate the model's performance.
# Tokenization and Padding: Convert the text reviews to sequences of integers and pad them to ensure uniform length.
# ------------------------------------------------------------------


# Define features and target variable
X = imdb_data['review']
y = imdb_data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization and padding
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = pad_sequences(X_train_sequences, maxlen=200, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=200, padding='post', truncating='post')


# 4. Build and Train the Model
# This section constructs and trains the deep learning model:

# Model Construction: Build a Sequential model with embedding, LSTM, and dense layers.
# Compilation: Compile the model with Adam optimizer and binary cross-entropy loss function. Print the model summary for verification.
# Model Training: Train the model using the padded training data, setting the number of epochs and batch size, and use the testing data for validation.
# ------------------------------------------------------------------

# Build the model
model = Sequential([
    Embedding(10000, 64, input_length=200),
    LSTM(64, return_sequences=True),
    Dropout(0.5),
    LSTM(32),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(X_train_padded, y_train, epochs=5, validation_data=(X_test_padded, y_test), batch_size=64)

# 5. Evaluate the Model
# This section evaluates the model's performance and visualizes the training process:

# Model Evaluation: Use the evaluate method to calculate loss and accuracy on the test set.
# Print Accuracy: Display the test accuracy to assess the model's performance.
# Plot Training History: Visualize the training and validation accuracy and loss over epochs to understand the model's learning process.
# ------------------------------------------------------------------

# Evaluate the model
loss, accuracy = model.evaluate(X_test_padded, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 6. Make Predictions
# This section creates a function to predict the sentiment of a given review:

# Predict Sentiment: Convert the review to a sequence, pad it, and use the model to predict the sentiment.
# Test Function: Demonstrate the function with a sample review to show how it works.
# ------------------------------------------------------------------


# Function to predict sentiment
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200, padding='post', truncating='post')
    prediction = model.predict(padded_sequence)
    return 'positive' if prediction >= 0.5 else 'negative'

# Test the function
sample_review = "This movie was fantastic! I really enjoyed it."
print(f"Review: {sample_review}")
print(f"Predicted Sentiment: {predict_sentiment(sample_review)}")


                                                                                                                                
