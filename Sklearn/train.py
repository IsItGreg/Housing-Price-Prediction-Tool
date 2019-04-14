# used to train the model and save the model weights

import argparse
import numpy as np
import pickle
import json
import os
import pandas as pd

from sklearn.linear_model import Lasso, SGDRegressor, ElasticNet

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
    df = pd.read_csv(filename)

    # get the targets
    y = train.pop("AV_TOTAL")

    #df.drop(["ST_NUM", ])

    array = df.values

    #sample data
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([4, 6])
    X_val = np.array([[1, 2], [3, 4], [5, 6]])
    y_val = np.array([4, 6, 8])
    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default="prop_dat_2019.csv",
                        type=str,
                        required=False,
                        help="the input dataset to be used to train the model")
    parser.add_argument("--output_dir",
                        default="test",
                        type=str,
                        required=False,
                        help="the output file for the ")
    parser.add_argument("--model_type",
                        default="ElasticNet",
                        type=str,
                        required=False,
                        help="the kind of model to use "
                        "[Lasso, SGDRegressor, ElasticNet]")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # load data into numpy array
    X_train, y_train, X_val, y_val = load_data(args.data_dir)

    # create model
    if args.model_type == "Lasso":
        # change the alpha value for shit
        model = Lasso(alpha=0.01, fit_intercept=True, normalize=True,
            precompute=False, copy_X=True, max_iter=1000, tol=0.0001,
            warm_start=False, positive=False, random_state=None,
            selection='cyclic')
    elif args.model_type == "SGDRegressor":
        model = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.001,
            l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None,
            shuffle=True, verbose=0, epsilon=0.1, random_state=None,
            learning_rate='invscaling', eta0=0.01, power_t=0.25,
            early_stopping=False, validation_fraction=0.1,
            n_iter_no_change=5, warm_start=False, average=False,
            n_iter=None)
    elif args.model_type == "ElasticNet":
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True,
            normalize=True, precompute=False, max_iter=1000, copy_X=True,
            tol=0.0001, warm_start=False, positive=False, random_state=None,
            selection='cyclic')

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