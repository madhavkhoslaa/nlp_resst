import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from flask_restful import Resource
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from flask_restful import reqparse
class classifier(Resource):
    parser = reqparse.RequestParser()
    parser.add_argument('Comment', type=int, help='The ratinh you want to classify')
    classify_this = parser.parse_args()
    dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
    nltk.download('stopwords')
    corpus = []
    for i in range(0, 1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    cv = CountVectorizer(max_features = 1500)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values
    classifier = GaussianNB()
    classifier.fit(X,y)
    result = classifier.predict(classify_this)
    def post(self):
        return {"Classified as":result}
