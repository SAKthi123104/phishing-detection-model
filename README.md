from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


emails = [
    "URGENT: Your account has been suspended! Verify your credentials immediately at http://secure-bank-login.com",
    "Hey, are we still meeting for lunch today at 1 PM?",
    "Congratulations! You won a $1000 Amazon gift card. Click here to claim your prize now.",
    "Attached is the project update report for Q2. Please review before the meeting.",
    "Dear customer, someone attempted to log into your account. Click this link to reset your password.",
    "Can you send me the notes from yesterday's history lecture?",
    "Your Netflix subscription payment failed. Update your credit card details here.",
    "Hi Mom, I will be coming home a bit late tonight. Don't wait up for dinner.",
    "Verify your bank details within 24 hours or your account will be permanently blocked.",
    "The manager wants you to review this document by the end of the day."
]

labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Phishing, 0 = Safe


X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("--- Model Training Complete ---")

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n--- Performance Metrics ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\n--- Confusion Matrix ---")
print(cm)
print("(Row 1: Actual Safe, Row 2: Actual Phishing)")
print("(Col 1: Predicted Safe, Col 2: Predicted Phishing)")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Safe', 'Phishing']))
