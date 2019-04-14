#!flask/bin/python

import sys

from flask import Flask, jsonify, render_template, request, redirect, Response
import random, json
from run_folder.run import run_regressor

app = Flask(__name__)


@app.route('/')
def output():
    # serve index template
    return render_template('housePrice.html')


@app.route('/receiver', methods=['GET'])
def worker():
    # read json + reply
    values = request.args.listvalues()
    listval = []
    for x in values:
        listval.append(int(x[0]))

    #  listval is list of numbers
    result = run_regressor(listval)

    return jsonify(result=result[0])


if __name__ == '__main__':
    # run!
    app.run()
