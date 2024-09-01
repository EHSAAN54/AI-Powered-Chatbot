import spacy
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    """Preprocess the input text: tokenization and lemmatization."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def get_sentiment(text):
    """Determine the sentiment of the input text (simplified)."""
    # Placeholder sentiment analysis; this can be expanded with more sophisticated methods.
    positive_words = {'good', 'great', 'happy', 'love', 'excellent'}
    negative_words = {'bad', 'sad', 'hate', 'poor', 'terrible'}

    tokens = word_tokenize(text.lower())
    positive_score = sum(word in positive_words for word in tokens)
    negative_score = sum(word in negative_words for word in tokens)

    if positive_score > negative_score:
        return 'positive'
    elif negative_score > positive_score:
        return 'negative'
    else:
        return 'neutral'
