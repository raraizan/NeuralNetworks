import os
import json

from flask import Flask, Response, render_template, request, url_for
import numpy
from flask_cors import CORS

import nn

DATASETS_PATH = 'flask_app/datasets'
MODELS_PATH = 'flask_app/models'

app = Flask(__name__)
CORS(app)

neural_network = nn.NeuralNetwork()

@app.route('/')
def index():
    neural_network.load('sigmoid_05_08_2019_06:34:05.json', path=MODELS_PATH)
    return 'al ready'


@app.route('/create', methods=['POST'])
def create_network():
    content = request.json

    sizes = tuple(int(x) for x in conten['sizes'])
    function = content['activation_function']
    neural_network.create(sizes)
    return ''


# @app.route('/save', methods=['POST'])
# def save_network():
#     content = request.json
#     filename = content['filename']
#     neural_network.save(filename, path=MODELS_PATH)
#     return ''


# @app.route('/load', methods=['POST'])
# def load_network():
#     content = request.json
#     filename = content['filename']
#     neural_network.load(path=MODELS_PATH)
#     return ''


@app.route('/evaluate', methods=['GET'])
def evaluate():
    content = request.json

    data_to_evaluate = numpy.array([[float(x)] for x in content['values']])
    result = neural_network.feedforward(data_to_evaluate)
    data = {
        'values': [str(x) for x in result],
        'result': str(nn.misc.get_category(result))
    }
    response = app.response_class(
        response=json.dumps(data),
        mimetype='application/json'
    )
    return response
