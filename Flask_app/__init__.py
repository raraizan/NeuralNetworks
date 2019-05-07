import os

from flask import Flask

from NN import Perceptron


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'
