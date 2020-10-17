import json
import joblib
import sklearn

import numpy as np
from attr import validate
from flask import Flask, request, jsonify, app
from utils import clean_text
from marshmallow import Schema, fields, ValidationError

models = {
    "bernoulli": {
        "count": joblib.load("models/bernoulli_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/bernoulli_naive_bayes_with_tfidf_vectorizer.joblib"),
    },

    "categorical": {
        "count": joblib.load("models/categorical_naive_bayes_with_count_vectorizer.joblib"),
    },
    "complement": {
        "count": joblib.load("models/complement_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/complement_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
    "gaussian": {
        "count": joblib.load("models/gaussian_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/gaussian_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
    "multinomial": {
        "count": joblib.load("models/multinomial_naive_bayes_with_count_vectorizer.joblib"),
        "tfidf": joblib.load("models/multinomial_naive_bayes_with_tfidf_vectorizer.joblib"),
    },
}

class PredictSchema(Schema):
    model = fields.string(required=True)
    vectorizer = fields.string(required=True)
    text = fields.string(required=True)

class PredictAllSchema(Schema):
    text = fields.string(required=True)

def Validation(schema_class, controller, request_data):
    #get request body from json
    schema = schema_class()

    try:
        #validate request body against sxhema data types
        result = schema.load(request_data)

    except ValidationError as err:
        #return a nice message if the validation fails
        return jsonify(err.messages), 400

    #converting request body back to json
    response_data = controller(result)

    return jsonify(response_data),200

def predict(parameters: dict)->str:
    #all the necessary parameters to select the right mode
    model = parameters.pop("model")
    vectorizer = parameters("vectorizer")
    text = parameters("text")

    if model == "categorical" and "vectorizer" == "tfidf":
        return jsonify(error="categorical does not work with tfidf vectorizer"), 400
    X = [text] #input
    naive_bayes_model = models[model][vectorizer]
    y = naive_bayes_model.predict(X) #prediction

    #the final response
    response = "postive" if y else "negative"

    return response

def predict_all(parameters: dict)->dict:
    text = parameters.pop("text")

    #the final response
    response = {}

    x = [text] #input
    for model in models:
        response[model] = {}

        for vectorizer in models[model]:
            y = models[model][vectorizer].predict(x)#predict
            response[model][vectorizer] = "positive" if y else "negative"

    return response


app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def predict_controller():
    return validate(PredictSchema, predict, request.json)


@app.route('/predict_all', methods=["POST"])
def predict_all_controller():
    return validate(PredictAllSchema, predict_all, request.json)


@app.route('/ping')
def ping():
    return 'pong'


if __name__ == '__main__':
    app.run()
