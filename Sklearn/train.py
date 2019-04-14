# used to train the model and save the model weights

import argparse
import numpy as np
import pickle
import json
import os
import pandas as pd

from sklearn.linear_model import Lasso, SGDRegressor, ElasticNet, LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler

from run import run_regressor


def load_data(filename, args, do_normalize=True, do_scale=True):
    '''
    takes in the filename of the csv dataset, returns numpy arrays for the
    inputs and labels for the train and dev set

    probably use pandas to load data, and k-fold cross validation to create
    def and test sets

    :param filename: (str) csv file with data
    :param do_normalize: (bool) normalize if true
    :return:
        X_train: numpy array (n_train_samples*n_features)
        y_train: numpy array (n_train_samples)
        X_dev: numpy array (n_dev_samples*n_features)
        y_dev: numpy array (n_dev_samples)
    '''
    df = pd.read_csv(filename)
    arrays = []
    arrays.append(np.array(df.pop("LAND_SF").values))
    arrays.append(np.array(df.pop("YR_BUILT").values))
    arrays.append(np.array(df.pop("NUM_FLOORS").values))
    arrays.append(np.array(df.pop("NUM_PARK").values))
    arrays.append(np.array(df.pop("NUM_RMS").values))
    arrays.append(np.array(df.pop("NUM_BDRMS").values))
    arrays.append(np.array(df.pop("NUM_FULL_BTH").values))
    arrays.append(np.array(df.pop("NUM_HALF_BTH").values))
    arrays.append(np.array(df.pop("ZIP_CODE").values))


    X = np.array(arrays)
    X = np.transpose(X)
    # get the targets
    y = df.pop("AV_TOTAL").values

    if do_normalize:
        X = normalize(X, norm='max')

    if do_scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        scaler_filename = os.path.join(args.output_dir, "scaler.sav")
        pickle.dump(scaler, open(scaler_filename, 'wb'))

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    # sample data
    # X_train = np.array([[1, 0, 0], [0, 1, 0]])
    # y_train = np.array([1, 4])
    # X_val = np.array([[1, 0, 0]])
    # y_val = np.array([1])
    return X_train, y_train, X_test, y_test


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir",
                        default="data/formatted.csv",
                        type=str,
                        required=False,
                        help="the input dataset to be used to train the model")
    parser.add_argument("--output_dir",
                        default="SGDRegressor_5",
                        type=str,
                        required=False,
                        help="the output file for the ")
    parser.add_argument("--model_type",
                        default="SGDRegressor",
                        type=str,
                        required=False,
                        help="the kind of model to use "
                        "[Lasso, SGDRegressor, ElasticNet, SVR, LinearRegression]")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # load data into numpy array
    X_train, y_train, X_val, y_val = load_data(args.data_dir, args)

    # create model
    if args.model_type == "Lasso":
        # change the alpha value for shit
        model = Lasso(alpha=.1, fit_intercept=True, normalize=True,
            precompute=False, copy_X=True, max_iter=100000, tol=0.000001,
            warm_start=False, positive=True, random_state=None,
            selection='cyclic')
    elif args.model_type == "SGDRegressor":
        model = SGDRegressor(loss='squared_epsilon_insensitive',
            penalty='elasticnet', alpha=0.1,
            l1_ratio=0.15, fit_intercept=True, max_iter=10000, tol=.00000001,
            shuffle=True, verbose=1, epsilon=0.1, random_state=None,
            learning_rate='optimal', eta0=0.001, power_t=0.25,
            early_stopping=False, validation_fraction=0.1,
            n_iter_no_change=100, warm_start=False, average=False,
            n_iter=None)
    elif args.model_type == "ElasticNet":
        model = ElasticNet(alpha=.000001, l1_ratio=0.5, fit_intercept=True,
            normalize=True, precompute=False, max_iter=10000, copy_X=True,
            tol=0.0001, warm_start=False, positive=True, random_state=None,
            selection='cyclic')
    elif args.model_type == "SVR":
        model = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0,
            tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200,
            verbose=False, max_iter=10000)
    elif args.model_type == "LinearRegression":
        model = LinearRegression(fit_intercept=True, normalize=False,
            copy_X=True, n_jobs=None)

    # train the model with the X, and y train numpy arrays
    model.fit(X_train, np.log(y_train+1))

    # get score with the X, and y dev numpy arrays
    test_score = model.score(X_val, np.log(y_val+1))
    train_score = model.score(X_train, np.log(y_train+1))
    print("train: {}, test: {}".format(train_score, test_score))

    # save_parameters
    parameters = model.get_params()
    with open(os.path.join(args.output_dir, "params.json"), "w") as fp:
        json.dump(parameters, fp)

    # save the model weights
    model_weights_filename = os.path.join(args.output_dir, "trained_model.sav")
    pickle.dump(model, open(model_weights_filename, 'wb'))

    # get outputs
    output = str()
    for prediction, label in zip(run_regressor(X_val, model_weights_filename), y_val):
        output+="{}, {}\n".format(prediction, label)

    # save scorem outputs
    with open(os.path.join(args.output_dir, "score.txt"), "w") as fp:
        fp.write("train score: {}, test score:{}".format(train_score, test_score))
        fp.write(output)


if __name__ == '__main__':
    main()