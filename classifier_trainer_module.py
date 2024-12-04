import os
import json
import time
import joblib
import pandas as pd
from preprocess_index import Preprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # Import tqdm for progress tracking

class TopicClassifier:
    def __init__(self, model_file='model/topic_classifier_model.joblib', vectorizer_file='model/topic_vectorizer.joblib'):
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file
        self.preprocessor = Preprocessor()
        self.vectorizer = TfidfVectorizer(max_features=60000)
        self.classifier = SVC(kernel='linear')

    def load_data(self, json_file, csv_file):
        # Load data from the JSON file
        with open(json_file, 'r') as file:
            data = json.load(file)

        texts = []
        labels = []
        
        # Add data from scraped_data.json
        for topic, documents in data.items():
            for doc in documents:
                combined_text = f"{doc['title']} {doc['summary']}"
                texts.append(" ".join(self.preprocessor.tokenizer(combined_text)))
                labels.append(topic)
        
        # Load data from the CSV file (general questions dataset)
        general_data = pd.read_csv(csv_file)
        for _, row in general_data.iterrows():
            question = row["question"]
            answer = row["answer"]
            general_text = f"{question} {answer}"
            texts.append(" ".join(self.preprocessor.tokenizer(general_text)))
            labels.append("General")  # All questions are labeled as 'General'
        
        return texts, labels

    def train(self, texts, labels, test_size=0.25, random_state=42):
        print("Training SVM Model........")
        
        # Vectorizing the text data
        X = self.vectorizer.fit_transform(texts)
        y = labels

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Use tqdm to track progress of training
        print("Training model...")
        for epoch in tqdm(range(1), desc="Training Epochs", unit="epoch"):
            self.classifier.fit(X_train, y_train)
        
        print("Model Training Done !!")
        
        # Predict on the test set and evaluate accuracy
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        time.sleep(2)
        print(f"Accuracy: {accuracy:.2f}")

        # Save the trained model and vectorizer
        self.save_model()
        self.save_vectorizer()

        return accuracy

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_file), exist_ok=True)
        joblib.dump(self.classifier, self.model_file)
        print(f"Model saved to {self.model_file}")

    def save_vectorizer(self):
        os.makedirs(os.path.dirname(self.vectorizer_file), exist_ok=True)
        joblib.dump(self.vectorizer, self.vectorizer_file)
        print(f"Vectorizer saved to {self.vectorizer_file}")

    def load_model(self):
        self.classifier = joblib.load(self.model_file)

    def load_vectorizer(self):
        self.vectorizer = joblib.load(self.vectorizer_file)

    def predict(self, texts):
        # Tokenize and transform the input texts
        processed_texts = [" ".join(self.preprocessor.tokenizer(text)) for text in texts]
        X = self.vectorizer.transform(processed_texts)
        
        # Predict the labels for the given texts
        predictions = self.classifier.predict(X)
        return predictions

# Main entry point for training the model
if __name__ == "__main__":
    # Define file paths
    json_file = 'data/scraped_data.json'
    csv_file = 'data/general_questions_dataset.csv'  # Path to your CSV file with general questions
    model_file = 'model/topic_classifier_model.joblib'
    vectorizer_file = 'model/topic_vectorizer.joblib'

    # Initialize the classifier
    classifier = TopicClassifier(model_file=model_file, vectorizer_file=vectorizer_file)
    
    # Load data from both the JSON and CSV files
    texts, labels = classifier.load_data(json_file, csv_file)
    
    # Train the model
    classifier.train(texts, labels)
