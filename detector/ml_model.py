import pandas as pd
import numpy as np
import re
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


class ScamDetector:
    """Machine learning model for detecting scam offers"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.model = MultinomialNB()
        self.is_trained = False
        self.classification_report_str = None
        # Try to load model from disk
        self.load_model()
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_and_prepare_data(self, csv_path='spam.csv'):
        """Load and prepare the spam dataset"""
        try:
            # Try different encodings to handle the CSV file
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    print(f"Trying to load CSV with {encoding} encoding...")
                    df = pd.read_csv(csv_path, encoding=encoding)
                    print(f"Successfully loaded CSV with {encoding} encoding!")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error with {encoding} encoding: {e}")
                    continue
            
            if df is None:
                raise Exception("Could not load CSV file with any encoding")
            
            # Rename columns for clarity
            df.columns = ['label', 'text'] + [f'col_{i}' for i in range(len(df.columns) - 2)]
            
            # Clean the data
            df = df.dropna(subset=['text'])
            df['text'] = df['text'].apply(self.preprocess_text)
            
            # Convert labels
            df['label'] = df['label'].map(lambda x: {'ham': 0, 'spam': 1}.get(x, -1))
            
            print(f"Loaded {len(df)} samples from {csv_path}")
            print(f"Label distribution: {df['label'].value_counts().to_dict()}")
            
            return df['text'], df['label']
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Falling back to dummy data...")
            # Return dummy data if CSV not found
            return self._create_dummy_data()
    
    def _create_dummy_data(self):
        """Create dummy training data if CSV is not available"""
        legitimate_texts = [
            "Hello, how are you doing today?",
            "Please confirm your appointment for tomorrow",
            "Your order has been shipped and will arrive soon",
            "Thank you for your purchase",
            "Meeting scheduled for 3 PM today",
            "Your account has been updated successfully",
            "Please review the attached document",
            "Happy birthday! Hope you have a great day",
            "The weather is nice today",
            "Looking forward to seeing you soon"
        ]
        
        scam_texts = [
            "CONGRATULATIONS! You've won $1000! Click here to claim",
            "URGENT: Your account has been suspended. Call now to verify",
            "FREE iPhone! Limited time offer. Text YES to claim",
            "You've been selected for a $5000 prize! Call immediately",
            "Your bank account needs verification. Click this link",
            "WINNER! You've won a luxury vacation. Claim now!",
            "URGENT: Your package is waiting. Click to track",
            "FREE money! You've been chosen for a cash prize",
            "Your computer has a virus. Call this number immediately",
            "CONGRATULATIONS! You're our lucky winner!"
        ]
        
        texts = legitimate_texts + scam_texts
        labels = [0] * len(legitimate_texts) + [1] * len(scam_texts)
        
        return texts, labels
    
    def train(self, csv_path='spam.csv', force_retrain=False):
        """Train the scam detection model"""
        if self.is_trained and not force_retrain:
            # If model is already trained, generate classification report from existing data
            if not self.classification_report_str:
                self._generate_classification_report(csv_path)
            return getattr(self, '_last_accuracy', 0.0)
        
        print("Loading and preparing data...")
        texts, labels = self.load_and_prepare_data(csv_path)
        
        print("Vectorizing text data...")
        X = self.vectorizer.fit_transform(texts)
        
        print("Splitting data for training...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        print("Training the model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self._last_accuracy = accuracy
        
        report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Scam'])
        self.classification_report_str = report
        
        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(report)
        
        self.is_trained = True
        self.save_model()
        return accuracy
    
    def _generate_classification_report(self, csv_path='spam.csv'):
        """Generate classification report from existing trained model"""
        try:
            print("Generating classification report from existing model...")
            texts, labels = self.load_and_prepare_data(csv_path)
            
            # Vectorize the data using the existing vectorizer
            X = self.vectorizer.transform(texts)
            
            # Split data for evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42
            )
            
            # Make predictions on test set
            y_pred = self.model.predict(X_test)
            
            # Generate classification report
            report = classification_report(y_test, y_pred, target_names=['Legitimate', 'Scam'])
            self.classification_report_str = report
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            self._last_accuracy = accuracy
            
            print(f"Classification report generated successfully!")
            print(f"Accuracy: {accuracy:.2%}")
            print("\nClassification Report:")
            print(report)
            
        except Exception as e:
            print(f"Error generating classification report: {e}")
            self.classification_report_str = "Error generating classification report"
    
    def get_classification_report(self):
        """Return the last classification report as a string"""
        return getattr(self, 'classification_report_str', None)
    
    def predict(self, text):
        """Predict if a text is a scam or legitimate"""
        if not self.is_trained:
            self.train()
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        # Vectorize the text
        text_vector = self.vectorizer.transform([processed_text])
        # Make prediction
        prediction = self.model.predict(text_vector)[0]
        probability = self.model.predict_proba(text_vector)[0]
        confidence = max(probability)
        return {
            'is_scam': bool(prediction),
            'confidence': confidence,
            'probability_legitimate': probability[0],
            'probability_scam': probability[1]
        }
    
    def save_model(self, filepath='scam_detector_model.pkl'):
        """Save the trained model"""
        if self.is_trained:
            # Ensure classification report is available before saving
            if not self.classification_report_str:
                self._generate_classification_report()
            
            model_data = {
                'vectorizer': self.vectorizer,
                'model': self.model,
                'is_trained': self.is_trained,
                'classification_report_str': self.classification_report_str,
                '_last_accuracy': getattr(self, '_last_accuracy', 0.0)
            }
            joblib.dump(model_data, filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='scam_detector_model.pkl'):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.vectorizer = model_data['vectorizer']
            self.model = model_data['model']
            self.is_trained = model_data['is_trained']
            self.classification_report_str = model_data.get('classification_report_str', None)
            self._last_accuracy = model_data.get('_last_accuracy', 0.0)
            print(f"Model loaded from {filepath}")
            return True
        return False


# Global instance
scam_detector = ScamDetector()

def get_scam_detector():
    """Get the global scam detector instance"""
    return scam_detector 