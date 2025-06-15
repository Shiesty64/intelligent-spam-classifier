import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 

nltk.download('stopwords')

try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df.rename(columns={'v1': 'label', 'v2': 'message'})
    df = df[['label', 'message']]
except FileNotFoundError:
    print("Error: 'spam.csv' not found. Please ensure it is in the same directory as your script.")
    print("You can find it on Kaggle: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset")
    exit() 

print("Original DataFrame head:")
print(df.head())
print("\nLabel Distribution:")
print(df['label'].value_counts())

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)


df['processed_message'] = df['message'].apply(preprocess_text)

print("\nDataFrame head after preprocessing:")
print(df[['message', 'processed_message']].head())

tfidf_vectorizer = TfidfVectorizer(max_features=5000)

X = tfidf_vectorizer.fit_transform(df['processed_message'])
y = df['label'] 

print(f"\nShape of TF-IDF feature matrix (X): {X.shape}") 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set shape: {X_train.shape}, {y_train.shape}")
print(f"Testing set shape: {X_test.shape}, {y_test.shape}")

model = MultinomialNB()

model.fit(X_train, y_train)

print("\nModel trained successfully!")

y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}") 
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


def predict_email(text, model, vectorizer):
    processed_text = preprocess_text(text)
    text_features = vectorizer.transform([processed_text])
    prediction = model.predict(text_features)[0] 
    prediction_proba = model.predict_proba(text_features)[0]
    return prediction, prediction_proba

print("\n--- Testing with New Messages ---")

# Example 1: Ham message
ham_message = "Hey, I'm just checking in to see how you're doing. Let's catch up soon!"
pred, proba = predict_email(ham_message, model, tfidf_vectorizer)
print(f"Message: '{ham_message}'")
print(f"Predicted as: {pred} (Probabilities: Ham={proba[0]:.4f}, Spam={proba[1]:.4f})")

# Example 2: Spam message
spam_message = "WIN a FREE iPhone now!!! Click this link to claim your prize! Limited time offer!"
pred, proba = predict_email(spam_message, model, tfidf_vectorizer)
print(f"\nMessage: '{spam_message}'")
print(f"Predicted as: {pred} (Probabilities: Ham={proba[0]:.4f}, Spam={proba[1]:.4f})")

# Example 3: Another ham
ham_message_2 = "Can we reschedule our meeting for tomorrow at 3 PM? I have a conflict."
pred, proba = predict_email(ham_message_2, model, tfidf_vectorizer)
print(f"\nMessage: '{ham_message_2}'")
print(f"Predicted as: {pred} (Probabilities: Ham={proba[0]:.4f}, Spam={proba[1]:.4f})")

# Example 4: Another spam
spam_message_2 = "URGENT! Your account has been suspended. Verify your details at http://suspicious-link.com to avoid closure."
pred, proba = predict_email(spam_message_2, model, tfidf_vectorizer)
print(f"\nMessage: '{spam_message_2}'")
print(f"Predicted as: {pred} (Probabilities: Ham={proba[0]:.4f}, Spam={proba[1]:.4f})")