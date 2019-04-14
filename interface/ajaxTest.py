#!flask/bin/python

import sys

from flask import Flask, jsonify, render_template, request, redirect, Response
import random, json
from run_folder.run import run_regressor
from geopy.geocoders import Nominatim

app = Flask(__name__)


@app.route('/')
def output():
    # serve index template
    return render_template('housePrice.html')


@app.route('/receiver', methods=['GET'])
def worker():
    # read json + reply
    values = request.args.listvalues()
    #geolocator = Nominatim(user_agent="thisisatest")
    listval = []
    addr = ''
    for i, x in enumerate(values):
        if i < 9:
            listval.append(int(x[0]))
        else:
            addr += x[0] + ' '

    #location = geolocator.geocode(addr)
    #listval.append(location.latitude)
    #listval.append(location.longitude)

    print(listval)


    #  listval is list of numbers
    result = run_regressor(listval)

    return jsonify(result=result[0])


if __name__ == '__main__':
    # run!
    app.run()
