from functools import wraps

import joblib
from flask import Flask, request, jsonify
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


def validate_json(schema_class):
    def decorator(f):
        @wraps(f)
        def decorator(*args, **kwargs):
            #get request body from json
            schema = schema_class()

            try:
                #validate request body against schema data types
                result = schema.load(request.json)
                return f(*args **kwargs)

            except ValidationError as err:
                #return a nice message if validation
                return jsonify(err.messages), 400
            return decorated_function
        return decorator


app = Flask(__name__)

@app.route('/predict', methods=["POST"])
@validate_json(schema_class=PredictAllSchema)
def predict_controller():
    parameters = request.json

    #all necessary parameters to select the right mode
    model = parameters.pop("model")
    vectorizer = parameters.pop("vectorization")
    text = parameters.pop("text")

    if model == "categorical" and vectorizer =="tfidf":
        return jsonify(error = "categorical does not work with tfidf vectorizer"), 400

    x = [text] #input text
    naive_bayes_model = models[model][vectorizer]
    y = naive_bayes_model.predict(x) #output

    #final response to send back
    response = "posiitive" if y else "negative"
    return response

@app.route('/predict', methods=["POST"])
@validate_json(schem_class = PredictAllSchema)
def predict_All():
    text = request.json.pop("text")

    response = {}

    x = [text] #input
    for model in models:
        response[model] = {}

        for vectorization in models[model]:
            y = models[model][vectorization].predict(x) #prediction

            response[model][vectorization] = "positive" if y else "negative"
        return response


@app.route('/ping')
def ping():
    return 'pong'


if __name__ == '__main__':
    app.run()
