import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("review.csv")

X = df["review"]
y = df["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_vector = vectorizer.fit_transform(X)
#train model
model=LogisticRegression()
model.fit(X_vector,y)
#save model and vectorizer
joblib.dump(model,"model.pkl")
joblib.dump(vectorizer,"vectorizer.pkl")
print("Model Trained successfully!")