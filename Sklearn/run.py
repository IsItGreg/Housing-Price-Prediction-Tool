# used to get predictions from the model
# simply takes in inputs and returns the results
# probably use flask, worry about this later
import numpy as np
import pickle

from sklearn.linear_model import Lasso, SGDRegressor, ElasticNet

MODEL_FILENAME = "Lasso_1/trained_model.sav"


def create_inputs(data):
    """
    recieves a ?????
    :param data:
    :return:
    numpy array[n_features] that has correct corresponding columns
    """
    inputs = data
    return inputs


def run_regressor(data):
    """
    recieves ??? and returns the prediction from the model
    :param data:
    :return:
    """

    # create inputs to classifier from json
    inputs = create_inputs(data)

    # load model
    model = pickle.load(open(MODEL_FILENAME, 'rb'))

    # get prediction
    output = model.predict(inputs)

    # return prediction
    return output
