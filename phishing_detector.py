"""
Phishing Email Detection Model using Machine Learning
Built by Nkul Suthar | Thiranex-Project-3

This model classifies emails as Phishing or Safe/Legitimate
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class PhishingDetector:
    """Phishing Email Detection Model - Developed by Nkul Suthar"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.model_name = "Random Forest"
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from email data"""
        
        print("\n🔍 Extracting features from emails...")
        
        features = pd.DataFrame()
        
        # 1. Email content features
        features['subject_length'] = df['subject'].str.len()
        features['body_length'] = df['body'].str.len()
        features['subject_word_count'] = df['subject'].str.split().str.len()
        features['body_word_count'] = df['body'].str.split().str.len()
        
        # 2. URL features
        features['has_url'] = df['has_url'].astype(int)
        features['url_count'] = df['url_count']
        features['has_suspicious_url'] = df['url_suspicious']
        
        # 3. Attachment features
        features['has_attachment'] = df['has_attachment'].astype(int)
        
        # 4. Linguistic features
        features['exclamation_count'] = df['exclamation_count']
        features['urgent_words'] = df['urgent_words']
        features['has_suspicious_words'] = df['has_suspicious_words']
        
        # 5. Special character counts
        features['question_marks'] = df['body'].str.count(r'\?')
        features['capital_letters'] = df['body'].str.count(r'[A-Z]')
        
        # 6. Percentage of capital letters
        features['capital_ratio'] = features['capital_letters'] / (features['body_length'] + 1)
        
        # 7. Number of links
        features['link_count'] = df['body'].str.count(r'http[s]?://')
        
        # 8. Presence of sensitive words
        sensitive_words = ['password', 'credit card', 'ssn', 'social security', 'bank account']
        features['has_sensitive_words'] = df['body'].str.lower().apply(
            lambda x: any(word in x for word in sensitive_words)
        ).astype(int)
        
        # 9. Subject features
        features['subject_has_alert'] = df['subject'].str.contains('alert|urgent|verify', case=False).astype(int)
        features['subject_all_caps'] = df['subject'].str.isupper().astype(int)
        
        print(f"✅ Extracted {len(features.columns)} features")
        print(f"   Features: {', '.join(features.columns[:10])}...")
        
        return features
    
    def extract_text_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract TF-IDF features from email text"""
        
        print("\n📝 Extracting text features using TF-IDF...")
        
        # Combine subject and body for text analysis
        df['combined_text'] = df['subject'] + " " + df['body']
        
        # TF-IDF Vectorization
        self.vectorizer = TfidfVectorizer(
            max_features=100,  # Top 100 important words
            stop_words='english',
            ngram_range=(1, 2)  # Unigrams and bigrams
        )
        
        text_features = self.vectorizer.fit_transform(df['combined_text'])
        
        # Get important words
        feature_names = self.vectorizer.get_feature_names_out()
        print(f"✅ Extracted {text_features.shape[1]} text features")
        print(f"   Top words: {', '.join(feature_names[:10])}")
        
        return text_features.toarray()
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare features and labels for training"""
        
        print("\n📊 Preparing data for training...")
        
        # Extract numeric features
        numeric_features = self.extract_features(df)
        
        # Extract text features
        text_features = self.extract_text_features(df)
        
        # Combine features
        X = np.hstack([numeric_features.values, text_features])
        
        # Labels
        y = df['label'].values
        
        print(f"✅ Final feature matrix shape: {X.shape}")
        print(f"✅ Labels shape: {y.shape}")
        print(f"   Positive (Phishing): {(y == 1).sum()}")
        print(f"   Negative (Safe): {(y == 0).sum()}")
        
        # Store feature names for interpretation
        self.feature_names = list(numeric_features.columns) + list(self.vectorizer.get_feature_names_out())
        
        return X, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, model_type: str = "random_forest"):
        """Train the model"""
        
        print(f"\n🤖 Training {model_type.upper()} model...")
        
        if model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            self.model_name = "Random Forest"
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
            self.model_name = "Logistic Regression"
        elif model_type == "naive_bayes":
            self.model = MultinomialNB()
            self.model_name = "Naive Bayes"
        else:
            raise ValueError("Unknown model type")
        
        self.model.fit(X_train, y_train)
        print(f"✅ Model trained successfully!")
        
        # Feature importance (for Random Forest)
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            top_indices = np.argsort(importance)[-10:][::-1]
            print(f"\n📊 Top 10 Important Features:")
            for idx in top_indices:
                if idx < len(self.feature_names):
                    print(f"   • {self.feature_names[idx]}: {importance[idx]:.4f}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model performance"""
        
        print(f"\n📈 Evaluating {self.model_name} Model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        # Display results
        print("\n" + "="*60)
        print("📊 MODEL PERFORMANCE METRICS")
        print("="*60)
        print(f"✅ Accuracy:  {accuracy*100:.2f}%")
        print(f"✅ Precision: {precision*100:.2f}%")
        print(f"✅ Recall:    {recall*100:.2f}%")
        print(f"✅ F1-Score:  {f1*100:.2f}%")
        
        print("\n📊 CONFUSION MATRIX:")
        print("                 Predicted")
        print("                 Safe  Phish")
        print(f"   Actual Safe   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"          Phish   {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        print("\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
        
        return results
    
    def predict_email(self, subject: str, body: str) -> dict:
        """Predict if a single email is phishing"""
        
        # Create DataFrame with single email
        email_df = pd.DataFrame([{
            'subject': subject,
            'body': body,
            'has_url': 1 if 'http' in body else 0,
            'url_count': body.count('http'),
            'has_attachment': 0,
            'exclamation_count': body.count('!'),
            'urgent_words': sum(word in body.lower() for word in ['urgent', 'immediate', 'verify']),
            'has_suspicious_words': 1 if any(word in body.lower() for word in ['verify', 'suspended', 'click here']) else 0,
            'url_suspicious': 1 if any(sus in body for sus in ['bit.ly', 'tinyurl', '.xyz', '.ru']) else 0,
            'label': 0  # Dummy label
        }])
        
        # Extract features
        numeric_features = self.extract_features(email_df)
        text_features = self.extract_text_features(email_df)
        
        # Combine features
        X = np.hstack([numeric_features.values, text_features])
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0] if hasattr(self.model, 'predict_proba') else [0, 0]
        
        result = {
            'is_phishing': bool(prediction),
            'confidence': float(max(probability)),
            'probability_safe': float(probability[0]),
            'probability_phishing': float(probability[1])
        }
        
        return result
    
    def save_model(self, filename: str = "phishing_model.pkl"):
        """Save trained model and vectorizer"""
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'model_name': self.model_name
        }, filename)
        print(f"\n💾 Model saved to {filename}")
    
    def load_model(self, filename: str = "phishing_model.pkl"):
        """Load trained model"""
        data = joblib.load(filename)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.feature_names = data['feature_names']
        self.model_name = data['model_name']
        print(f"\n📂 Model loaded from {filename}")

def main():
    print("="*60)
    print("🔐 PHISHING EMAIL DETECTION MODEL")
    print("Built by Nkul Suthar | Thiranex-Project-3")
    print("="*60)
    
    # Step 1: Load or generate dataset
    print("\n📂 Loading dataset...")
    try:
        df = pd.read_csv('phishing_emails.csv')
        print(f"✅ Loaded {len(df)} emails from file")
    except FileNotFoundError:
        print("⚠️ Dataset not found! Generating new dataset...")
        from generate_dataset import EmailDatasetGenerator
        generator = EmailDatasetGenerator()
        df = generator.generate_dataset(2000)
        generator.save_dataset(df)
    
    # Step 2: Prepare data
    detector = PhishingDetector()
    X, y = detector.prepare_data(df)
    
    # Step 3: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Testing set: {len(X_test)} samples")
    
    # Step 4: Train multiple models
    models = ['random_forest', 'logistic_regression', 'naive_bayes']
    best_model = None
    best_accuracy = 0
    
    for model_type in models:
        detector = PhishingDetector()
        detector.train(X_train, y_train, model_type)
        results = detector.evaluate(X_test, y_test)
        
        if results['accuracy'] > best_accuracy:
            best_accuracy = results['accuracy']
            best_model = detector
    
    # Step 5: Save best model
    print("\n" + "="*60)
    print(f"🏆 BEST MODEL: {best_model.model_name}")
    print(f"   Accuracy: {best_accuracy*100:.2f}%")
    print("="*60)
    
    best_model.save_model()
    
    # Step 6: Test on sample emails
    print("\n🔍 TESTING ON SAMPLE EMAILS:")
    print("="*60)
    
    test_emails = [
        {
            "subject": "URGENT: Your PayPal Account Has Been Suspended",
            "body": "Dear Customer, We detected unusual activity. Verify now: http://bit.ly/paypal-verify"
        },
        {
            "subject": "Meeting Tomorrow at 2 PM",
            "body": "Hi team, Let's discuss the project tomorrow. Here's the link: https://meet.google.com/abc-def"
        },
        {
            "subject": "Your Amazon Order #12345",
            "body": "Your order has been shipped. Track here: https://amazon.com/track"
        },
        {
            "subject": "WINNER! You've Won $1,000,000",
            "body": "CONGRATULATIONS! Click here to claim your prize: http://prize-winner.xyz/claim"
        }
    ]
    
    for i, email in enumerate(test_emails, 1):
        result = best_model.predict_email(email['subject'], email['body'])
        
        print(f"\n📧 Test {i}:")
        print(f"   Subject: {email['subject'][:50]}...")
        
        if result['is_phishing']:
            print(f"   🔴 RESULT: PHISHING DETECTED!")
            print(f"   Confidence: {result['confidence']*100:.1f}%")
        else:
            print(f"   🟢 RESULT: SAFE EMAIL")
            print(f"   Confidence: {result['confidence']*100:.1f}%")

if __name__ == "__main__":
    main()
