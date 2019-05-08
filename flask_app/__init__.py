import os

from flask import Flask

import nn

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'