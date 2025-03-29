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


#Access data
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')
y_train = pd.read_csv('data/processed/y_train.csv')
y_test = pd.read_csv('data/processed/y_test.csv')
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

#cross validation
cv=5
# Define the model
regressor = RandomForestRegressor(random_state=42)

# Define the hyperparameter grid
params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
def grid_search(regressor,params,cv):

    # Perform grid search
    grid_search = GridSearchCV(estimator=regressor, param_grid=params, cv=cv, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Display the best hyperparameters
    best_params=grid_search.best_params_
    print("Best parameters found:", best_params)

    # Evaluate the model on test data
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error on test data:", mse)

    return best_params


def save_parameters(parameters, output_folderpath):
    # Save dataframes to their respective output file paths
    for params, filename in zip([parameters], ['best_params.pkl']):
        output_filepath = os.path.join(output_folderpath, filename)
        if check_existing_file(output_filepath):
            with open(output_filepath, 'wb') as file:
                pickle.dump(params, file)


def save_model(model, output_folderpath):
    # Save dataframes to their respective output file paths
    for model, filename in zip([model], ['random_forest_regressor.pkl']):
        output_filepath = os.path.join(output_folderpath, filename)
        if check_existing_file(output_filepath):
            with open(output_filepath, 'wb') as file:
                pickle.dump(model, file)
            

if __name__ == '__main__':
    # Define the model
    regressor = RandomForestRegressor(random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
}
    best_model,best_params=grid_search(regressor,param_grid,cv)
    save_parameters(best_params,"models")
    save_model(best_model,"models")



