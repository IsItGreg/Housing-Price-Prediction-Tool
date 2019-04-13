# used to train the model and save the model weights

import argparse
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
import json
import os

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
    X_train = np.array([[3, 4], [1, 2]])
    y_train = np.array([2, 4])
    X_val = np.array([[3, 4], [1, 2]])
    y_val = np.array([2, 4])
    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default="data_dir",
                        type=str,
                        required=False,
                        help="the input dataset to be used to train the model")
    parser.add_argument("--output_dir",
                        default="test",
                        type=str,
                        required=False,
                        help="the output file for the ")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # load data into numpy array
    X_train, y_train, X_val, y_val = load_data(args.data_dir)

    # create model
    model = LinearRegression(normalize=True)

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
    pickle.dump(model, open(os.path.join(args.output_dir, "trained_model"), 'wb'))







if __name__ == '__main__':
    main()