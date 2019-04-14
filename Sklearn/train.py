# used to train the model and save the model weights

import argparse
import numpy as np
import pickle
import json
import os

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

from run import run_regressor

def load_data(filename):
    '''
    takes in the filename of the csv dataset, returns numpy arrays for the
    inputs and labels for the train and dev set

    probably use pandas to load data, and k-fold cross validation to create
    def and test sets

    :param filename: (str) csv file with data
    :return:
        X_train: numpy array (n_train_samples*n_features)
        y_train: numpy array (n_train_samples)
        X_dev: numpy array (n_dev_samples*n_features)
        y_dev: numpy array (n_dev_samples)
    '''

    #sample data
    X_train = np.array([[3, 4], [1, 2], [5, 6]])
    y_train = np.array([5, 3, 7])
    X_val = np.array([[3, 4], [1, 2]])
    y_val = np.array([5, 3])
    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default="data_dir",
                        type=str,
                        required=False,
                        help="the input dataset to be used to train the model")
    parser.add_argument("--output_dir",
                        default="test_mlp",
                        type=str,
                        required=False,
                        help="the output file for the ")
    parser.add_argument("--model_type",
                        default="neural_network",
                        type=str,
                        required=False,
                        help="the kind of model to use [logistic_regression, "
                        "neural_network]")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # load data into numpy array
    X_train, y_train, X_val, y_val = load_data(args.data_dir)

    # create model
    if args.model_type == "logistic_regression":
        model = LinearRegression(normalize=True)
    elif args.model_type == "neural_network":
        model = MLPRegressor(max_iter=5000, early_stopping=False)

    # train the model with the X, and y train numpy arrays
    model.fit(X_train, y_train)

    # get score with the X, and y dev numpy arrays
    score = model.score(X_val, y_val)

    # save score
    with open(os.path.join(args.output_dir, "score.txt"), "w") as fp:
        fp.write("score: {}".format(score))

    # save_parameters
    parameters = model.get_params()
    with open(os.path.join(args.output_dir, "params.json"), "w") as fp:
        json.dump(parameters, fp)

    # save the model weights
    pickle.dump(model, open(os.path.join(
        args.output_dir, "trained_model.sav"), 'wb'))

    print(run_regressor("ass"))


if __name__ == '__main__':
    main()