import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nlp_processing import preprocess_text, get_sentiment

# Define chatbot model
class Chatbot:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model = self.build_model()

    def build_model(self):
        """Build a simple feedforward neural network model."""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(None,)),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, texts, labels):
        """Train the chatbot model with given texts and labels."""
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, padding='post')

        self.model.fit(padded_sequences, labels, epochs=5)

    def predict(self, text):
        """Predict response based on input text."""
        processed_text = preprocess_text(text)
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        padded_sequence = pad_sequences(sequence, padding='post')

        prediction = self.model.predict(padded_sequence)
        return 'response'  # Placeholder for actual response generation logic

    def get_sentiment(self, text):
        """Get sentiment of the input text."""
        return get_sentiment(text)
