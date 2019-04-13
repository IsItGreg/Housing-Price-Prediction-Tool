# used to train the model and save the model weights

import argparse
from sklearn.linear_model import LinearRegression
import numpy as np

def load_data(filename):
    '''
    takes in the filename of the csv dataset, returns numpy arrays

    :param filename:
    :return:
    '''
    X_train = np.array([[3, 4], [1, 2]])
    y_train = np.array([2, 4])
    X_val = np.array([[3, 4], [1, 2]])
    y_val = np.array([2, 4])
    return X_train, y_train, X_val, y_val

def main():
    parser = argparse.ArgumentParser

    parser.add_argument("--data_dir",
                        default="data_dir",
                        type=str,
                        required=False,
                        help="the input dataset to be used to train the model")
    parser.add_argument("--model_dir",
                        default="d",
                        type=str,
                        required=False,
                        help="the input dataset to be used to train the model")


    args = parser.parse_args()

    # load data into numpy array
    X_train, y_train, X_val, y_val = load_data(args.data_filename)

    # create model
    model = LinearRegression(normalize=True)

    # train the model with the x, and y numpy arrays
    model.fit(X_train, y_train)

    # save the model
    # weights = model.




if __name__ == '__main__':
    main()