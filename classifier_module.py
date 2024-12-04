import torch
from transformers import AutoTokenizer
import joblib
from preprocess_index import Preprocessor

class Classifier:
    def __init__(self, model_file='model/topic_classifier_model.joblib', vectorizer_file='model/topic_vectorizer.joblib'):
        # Load the saved SVM model and vectorizer
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file
        self.preprocessor = Preprocessor()  # Preprocessor to handle tokenization
        
        # Load the SVM model and vectorizer
        self.classifier = joblib.load(self.model_file)
        self.vectorizer = joblib.load(self.vectorizer_file)
        
        # List of topics
        self.topics = [
            "Economy", "Education", "Entertainment", "Environment",
            "Food", "Health", "Politics", "Sports",
            "Technology", "Travel", "General"
        ]

    def classify(self, query):
        # Tokenize and vectorize the input query
        processed_query = " ".join(self.preprocessor.tokenizer(query))
        X = self.vectorizer.transform([processed_query])

        # Predict the topic using the SVM classifier
        predicted_topic = self.classifier.predict(X)[0]
        confidence = self.get_confidence(X, predicted_topic)

        print(f"Predicted Topic: {predicted_topic}")
        return predicted_topic

    def get_confidence(self, X, predicted_topic):
        # SVM doesn't provide probabilities by default, but we can estimate confidence
        # using the decision function. The further the score from 0, the higher the confidence.
        decision_function = self.classifier.decision_function(X)
        topic_idx = self.topics.index(predicted_topic)
        confidence = abs(decision_function[0][topic_idx])
        return confidence


# Wrapper function for app.py
def classify_query(query, model_file='model/topic_classifier_model.joblib', vectorizer_file='model/topic_vectorizer.joblib'):
    classifier = Classifier(model_file=model_file, vectorizer_file=vectorizer_file)
    predicted_topic= classifier.classify(query)
    return predicted_topic


if __name__ == "__main__":
    query = "Tell me a joke"
    predicted_topic = classify_query(query)
    print(f"Predicted Topic: {predicted_topic}")