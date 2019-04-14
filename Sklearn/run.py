# used to get predictions from the model
# simply takes in inputs and returns the results
# probably use flask, worry about this later
import numpy as np
import pickle


def create_inputs(data):
    """
    recieves a ?????
    :param data:
    :return:
    numpy array[n_features] that has correct corresponding columns
    """
    inputs = np.array(data)
    #return inputs
    return data


def run_regressor(data, model_filename):
    """
    recieves np array of validation data, returns predictions for them
    :param data:
    :param model_filename: (str) filename of model to load
    :return:
    """

    # create inputs to classifier from json
    inputs = create_inputs(data)

    # load model
    model = pickle.load(open(model_filename, 'rb'))

    # get prediction
    output = np.exp(model.predict(inputs))-1

    # return prediction
    return output
