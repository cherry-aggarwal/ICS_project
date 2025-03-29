import numpy as np
import random

# Sample corpus (list of sentences)
corpus = [
    "hello world",
    "good morning",
    "have a great day",
    "see you soon",
    "good night",
    "I love programming",
    "we love coding",
    "she is my friend",
    "where is the nearest restaurant",
    "how are you"
]

# Tokenize corpus into words
words = set()
for sentence in corpus:
    for word in sentence.split():
        words.add(word)

# Create vocabulary and word-to-index mapping
word_to_index = {word: i for i, word in enumerate(words)}
index_to_word = {i: word for word, i in word_to_index.items()}
VOCAB_SIZE = len(word_to_index)

print("Vocabulary:", word_to_index)

# Generate training data using Skip-gram
WINDOW_SIZE = 2  # Context window size

def generate_training_data(corpus):
    training_data = []
    
    for sentence in corpus:
        words = sentence.split()
        for i, target_word in enumerate(words):
            target_idx = word_to_index[target_word]
            for j in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
                if j != 0 and 0 <= i + j < len(words):
                    context_idx = word_to_index[words[i + j]]
                    training_data.append((target_idx, context_idx))
    
    return training_data

training_data = generate_training_data(corpus)
print("Training data sample:", training_data[:5])  # Print first 5 samples

# Hyperparameters
EMBEDDING_DIM = 10  # Size of word vectors
LEARNING_RATE = 0.01
EPOCHS = 10000

# Initialize weights
W1 = np.random.uniform(-1, 1, (VOCAB_SIZE, EMBEDDING_DIM))  # Input-to-hidden weights
W2 = np.random.uniform(-1, 1, (EMBEDDING_DIM, VOCAB_SIZE))  # Hidden-to-output weights

# One-hot encoding function
def one_hot_vector(word_idx):
    one_hot = np.zeros(VOCAB_SIZE)
    one_hot[word_idx] = 1
    return one_hot

# Softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for target_idx, context_idx in training_data:
        # Forward pass
        target_vector = one_hot_vector(target_idx)  # One-hot encoded target word
        hidden_layer = np.dot(target_vector, W1)  # Compute hidden layer
        output_layer = np.dot(hidden_layer, W2)  # Compute scores
        y_pred = softmax(output_layer)  # Apply softmax

        # Compute loss (cross-entropy)
        loss = -np.log(y_pred[context_idx])
        total_loss += loss

        # Backpropagation
        y_pred[context_idx] -= 1  # Compute error
        dW2 = np.outer(hidden_layer, y_pred)  # Gradient for W2
        dW1 = np.outer(target_vector, np.dot(W2, y_pred))  # Gradient for W1

        # Update weights
        W1 -= LEARNING_RATE * dW1
        W2 -= LEARNING_RATE * dW2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {total_loss:.4f}")

print("Training complete!")

def get_word_vector(word):
    return W1[word_to_index[word]]

import pandas as pd
english_words = list(word_to_index.keys())
english_vectors = W1  # Your trained word vectors
df_en = pd.DataFrame(english_vectors, index=english_words)
df_en.to_csv("word2vec_en.csv", header=False)
