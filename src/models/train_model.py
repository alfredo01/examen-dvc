import numpy as np
import pandas as pd
import sys
import os
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))
from check_structure import check_existing_file, check_existing_folder

def load_params():
    #Load best params
    # open a file, where you stored the pickled data
    best_params_pkl = open('models/best_params.pkl', 'rb')
    best_params = pickle.load(best_params_pkl)
    best_params_pkl.close()
    return best_params


def train_model(X_train,y_train,params:dict):
    # Define the model
    regressor = RandomForestRegressor(**params,random_state=42)
    #--Train the model
    regressor.fit(X_train, y_train)
    return regressor


def save_model(model, output_folderpath):
    # Save dataframes to their respective output file paths
    for model, filename in zip([model], ['random_forest_regressor.pkl']):
        output_filepath = os.path.join(output_folderpath, filename)
        if check_existing_file(output_filepath):
            with open(output_filepath, 'wb') as file:
                pickle.dump(model, file)

if __name__ == '__main__':
    #Access data
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_train = np.ravel(y_train)

    params=load_params()
    rf_model=train_model(X_train,y_train,params)
    save_model(rf_model, "models")



