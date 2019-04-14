#!flask/bin/python

import sys

from flask import Flask, jsonify, render_template, request, redirect, Response
import random, json

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

    print(listval)
    result = 101010101
    return jsonify(result=result)


if __name__ == '__main__':
    # run!
    app.run()
