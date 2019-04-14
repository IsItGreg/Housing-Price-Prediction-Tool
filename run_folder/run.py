# used to get predictions from the model
# simply takes in inputs and returns the results
# probably use flask, worry about this later
import numpy as np
import pickle

from sklearn.linear_model import Lasso, SGDRegressor, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler


MODEL_FILENAME = "run_folder/SGDRegressor_4/trained_model.sav"


def create_inputs(data, do_normalize=True, do_scale=True):
    """
    recieves a list of model inputs
    :param data: (list[10]) containing model inputs
    :return:
    numpy array[n_features] that has correct corresponding columns
    """
    inputs = np.array([data])

    if do_normalize:
        inputs = normalize(inputs, norm='max')

    if do_scale:
        scaler = StandardScaler()
        scaler.fit(inputs)
        inputs = scaler.transform(inputs)

    return inputs


def run_regressor(data):
    """
    recieves a list of the data to be input into thr model
    :param data: list of inputs
    :return:
        the output of the model
    """

    # create inputs to classifier from json
    inputs = create_inputs(data)

    # load model
    model = pickle.load(open(MODEL_FILENAME, 'rb'))

    # get prediction
    output = np.exp(model.predict(inputs))-1

    # return prediction
    return output
