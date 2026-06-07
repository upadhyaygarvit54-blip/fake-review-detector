from flask import Flask,render_template,request
import joblib
from textblob import TextBlob

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

spam_words = [
    "buy now",
    "limited offer",
    "click here",
    "best product ever",
    "must buy"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    review = request.form['review']

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)

    probability = model.predict_proba(review_vector)

    confidence = round(max(probability[0])*100,2)

    # Sentiment analysis
    polarity = TextBlob(review).sentiment.polarity

    if polarity > 0:
        sentiment = "Positive 😊"
    elif polarity < 0:
        sentiment = "Negative 😞"
    else:
        sentiment = "Neutral 😐"

    found_words=[]

    for word in spam_words:
        if word.lower() in review.lower():
            found_words.append(word)

    if prediction[0]==1:
        result="⚠ Fake Review"
    else:
        result="✅ Genuine Review"

    return render_template(
        'index.html',
        prediction=result,
        confidence=confidence,
        sentiment=sentiment,
        words=found_words
    )

if __name__=="__main__":
    app.run(debug=True)