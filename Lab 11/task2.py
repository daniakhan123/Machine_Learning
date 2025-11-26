import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
data = pd.read_csv("product_reviews.csv")  # replace with your CSV filename

# 2. Preprocessing function
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    return " ".join(words)

# 3. Clean review text
data["clean_text"] = data["reviews.text"].apply(preprocess)
data = data[data["clean_text"].str.strip() != ""]

# 4. Map numerical ratings to sentiment
# Example: 1-2 = negative, 3 = neutral, 4-5 = positive
def map_rating_to_sentiment(r):
    if r <= 2:
        return "negative"
    elif r == 3:
        return "neutral"
    else:
        return "positive"

data["sentiment"] = data["reviews.rating"].apply(map_rating_to_sentiment)

# 5. Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(data["clean_text"])
y = data["sentiment"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# 8. Predict and evaluate
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

