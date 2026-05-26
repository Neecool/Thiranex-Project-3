"""
Generate Synthetic Phishing Email Dataset
Built by Nkul Suthar | Thiranex-Project-3
"""

import pandas as pd
import random
import csv
from datetime import datetime

class EmailDatasetGenerator:
    """Generate realistic email dataset for phishing detection"""
    
    def __init__(self):
        # Phishing keywords and patterns
        self.phishing_keywords = [
            'verify', 'account', 'suspended', 'urgent', 'security', 'alert',
            'unusual activity', 'login', 'password', 'update', 'confirm',
            'bank', 'paypal', 'amazon', 'apple', 'microsoft', 'google',
            'click here', 'link below', 'immediate action', 'limited time',
            'verify your identity', 'unauthorized access', 'locked account',
            'payment failed', 'refund', 'lottery', 'prize', 'winner'
        ]
        
        # Legitimate keywords
        self.legitimate_keywords = [
            'meeting', 'project', 'update', 'newsletter', 'welcome',
            'invoice', 'receipt', 'order confirmation', 'shipping',
            'your request', 'team', 'colleague', 'friend', 'family',
            'weekly report', 'summary', 'attached', 'document'
        ]
        
        # Suspicious URL patterns
        self.phishing_urls = [
            'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 'is.gd',
            'secure-login.xyz', 'account-verify.net', 'paypal-security.com',
            'amazon-verification.org', 'appleid-support.ru', 'microsoft-update.biz'
        ]
        
        # Legitimate URLs
        self.legitimate_urls = [
            'google.com', 'microsoft.com', 'amazon.com', 'paypal.com',
            'apple.com', 'github.com', 'linkedin.com', 'twitter.com'
        ]
        
        # Suspicious senders
        self.phishing_senders = [
            'security@paypal-verify.com', 'alert@amazon-security.net',
            'support@appleid-confirm.ru', 'admin@microsoft-update.biz',
            'noreply@bank-verify.org', 'service@paypal-security.com'
        ]
        
        # Legitimate senders
        self.legitimate_senders = [
            'noreply@google.com', 'support@microsoft.com', 'orders@amazon.com',
            'service@paypal.com', 'newsletter@github.com', 'team@linkedin.com'
        ]

    def generate_phishing_email(self) -> dict:
        """Generate a single phishing email"""
        
        # Random phishing content
        keyword = random.choice(self.phishing_keywords)
        url = random.choice(self.phishing_urls)
        sender = random.choice(self.phishing_senders)
        
        # Build email body
        body_templates = [
            f"Dear User,\n\nWe detected {keyword} on your account. Please verify immediately: http://{url}/verify\n\nFailure to verify will result in account suspension.",
            f"Attention!\n\nYour account has been {keyword}. Click the link below to restore access:\nhttp://{url}/secure-login\n\nThis is an automated message.",
            f"Security Alert!\n\nUnusual {keyword} detected. Confirm your identity now:\nhttp://{url}/confirm\n\nThank you.",
            f"Urgent: Account {keyword}\n\nWe need you to {keyword} your information. Click here: http://{url}/update\n\nDo not ignore this message.",
            f"Dear Customer,\n\nYour account has been temporarily {keyword} due to suspicious activity. Please verify: http://{url}/security-check"
        ]
        
        subject_templates = [
            f"Urgent: Account {keyword.upper()} Required",
            f"Security Alert: {keyword.title()} Detected",
            f"Action Required: {keyword.title()} Your Account",
            f"⚠️ Immediate {keyword.upper()} Needed",
            f"Account {keyword.upper()} Notification"
        ]
        
        email = {
            'subject': random.choice(subject_templates),
            'sender': sender,
            'body': random.choice(body_templates),
            'has_url': True,
            'url_count': random.randint(1, 3),
            'has_attachment': random.choice([True, False]),
            'exclamation_count': random.randint(2, 8),
            'urgent_words': random.randint(2, 6),
            'label': 1  # 1 = Phishing
        }
        
        return email

    def generate_legitimate_email(self) -> dict:
        """Generate a single legitimate email"""
        
        keyword = random.choice(self.legitimate_keywords)
        url = random.choice(self.legitimate_urls)
        sender = random.choice(self.legitimate_senders)
        
        body_templates = [
            f"Hi,\n\nHere's the {keyword} you requested. Check it out: https://{url}/docs\n\nBest regards,\nTeam",
            f"Hello,\n\nYour {keyword} has been processed. View details: https://{url}/account\n\nThanks!",
            f"Dear User,\n\n{keyword.title()} report attached. Let me know if you have questions.\n\nBest,\nColleague",
            f"Meeting {keyword}: Tomorrow at 2 PM. Here's the link: https://{url}/meeting\n\nSee you there!",
            f"Newsletter #{random.randint(1, 100)}: {keyword.title()} updates inside. Read more: https://{url}/news"
        ]
        
        subject_templates = [
            f"{keyword.title()} Update",
            f"Your {keyword} is ready",
            f"{keyword.title()} Report - {datetime.now().strftime('%B %Y')}",
            f"Meeting: {keyword.title()} Discussion",
            f"Newsletter: {keyword.title()} Edition"
        ]
        
        email = {
            'subject': random.choice(subject_templates),
            'sender': sender,
            'body': random.choice(body_templates),
            'has_url': random.choice([True, False]),
            'url_count': random.randint(0, 1),
            'has_attachment': random.choice([True, False]),
            'exclamation_count': random.randint(0, 2),
            'urgent_words': random.randint(0, 1),
            'label': 0  # 0 = Legitimate/Safe
        }
        
        return email

    def generate_dataset(self, num_samples: int = 2000) -> pd.DataFrame:
        """Generate complete dataset"""
        
        emails = []
        phishing_count = num_samples // 2
        legitimate_count = num_samples - phishing_count
        
        print(f"Generating {phishing_count} phishing emails...")
        for _ in range(phishing_count):
            emails.append(self.generate_phishing_email())
        
        print(f"Generating {legitimate_count} legitimate emails...")
        for _ in range(legitimate_count):
            emails.append(self.generate_legitimate_email())
        
        # Shuffle the dataset
        random.shuffle(emails)
        
        df = pd.DataFrame(emails)
        
        # Extract additional features
        df['subject_length'] = df['subject'].str.len()
        df['body_length'] = df['body'].str.len()
        df['has_suspicious_words'] = df['body'].apply(
            lambda x: any(word in x.lower() for word in ['verify', 'suspended', 'urgent', 'click here'])
        ).astype(int)
        
        df['url_suspicious'] = df.apply(
            lambda x: 1 if x['has_url'] and any(sus in x['body'] for sus in ['bit.ly', 'tinyurl', '.xyz', '.ru', '.biz']) else 0,
            axis=1
        )
        
        return df

    def save_dataset(self, df: pd.DataFrame, filename: str = "phishing_emails.csv"):
        """Save dataset to CSV"""
        df.to_csv(filename, index=False)
        print(f"\n✅ Dataset saved to {filename}")
        print(f"📊 Total emails: {len(df)}")
        print(f"   - Phishing: {(df['label'] == 1).sum()}")
        print(f"   - Legitimate: {(df['label'] == 0).sum()}")
        
        # Display sample
        print("\n📧 Sample Data:")
        print(df.head(10).to_string())
        
        return filename

def main():
    print("="*60)
    print("🔐 PHISHING EMAIL DATASET GENERATOR")
    print("Built by Nkul Suthar | Thiranex-Project-3")
    print("="*60)
    
    generator = EmailDatasetGenerator()
    
    # Generate dataset
    df = generator.generate_dataset(2000)  # 2000 emails
    generator.save_dataset(df)
    
    print("\n✅ Dataset ready for training!")
    print("\nNext step: Run phishing_detector.py to train model")

if __name__ == "__main__":
    main()
