import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def vectorize_texts(texts, method='tfidf'):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2, preprocessor=preprocess_text)
    else:
        vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2, preprocessor=preprocess_text)
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer
