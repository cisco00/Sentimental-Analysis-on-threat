import nltk
import string

from sklearn.base import TransformerMixin
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()

def clean_text(text: str)->str:
 #removing upper case
    text = text.lower()

    #removing puntuation
    for char in string.punctuation:
        text = text.replace(char, "")

    #lemmatize the words and join back into string text
    text = " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(text)])
    return text

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def predict(self, X, y=None, **fit_params):
        return self.X.todense()

    def _self_(self):
        return "DenseTransformer()"

    def __repr__(self):
        return self .__str__()

