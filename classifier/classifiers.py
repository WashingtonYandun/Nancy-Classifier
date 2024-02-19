from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np

class TextClassifier:
    def classify(self, text):
        raise NotImplementedError

class NaiveBayesClassifier(TextClassifier):
    def __init__(self, words, categories):
        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(words)
        self.clf = MultinomialNB()
        self.clf.fit(X, categories)

    def classify(self, text):
        title_vector = self.vectorizer.transform([text])
        probabilities = self.clf.predict_proba(title_vector)[0]
        all_categories = self.clf.classes_

        top_categories = [(cat, probabilities[idx]) for idx, cat in enumerate(all_categories)]
        top_categories.sort(key=lambda x: x[1], reverse=True)

        response = {
            "category": top_categories[0][0],
            "matches": [{"category": cat, "probability": prob} for cat, prob in top_categories[:3]]
        }
        return response
