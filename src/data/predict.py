import pandas as pd
import sys
import json
import pickle


def load_model():
    #Load best params
    # open a file, where you stored the pickled data
    model_pkl = open('models/random_forest_regressor.pkl', 'rb')
    model = pickle.load(model_pkl)
    model_pkl.close()
    return model

def model_predict(X_test,loaded_model):    
    predictions = loaded_model.predict(X_test)
    return predictions

def model_score(X_test,y_test,loaded_model):    
    score = loaded_model.score(X_test,y_test)
    return score


def save_predictions(predictions):
    predictions_df=pd.DataFrame(predictions)
    predictions_df.to_csv("data/predictions.csv")


def save_score(score):
    out_file = open("metrics/scores.json", "w")
    #json.loads()
    json.dump(score, out_file, indent = 6)
    out_file.close()

if __name__ == '__main__':
    X_test = pd.read_csv("data/processed/X_test.csv").values
    y_test = pd.read_csv("data/processed/y_test.csv").values
    rf_model=load_model()
    predictions=model_predict(X_test,rf_model)
    save_predictions(predictions)
    score=model_score(X_test,y_test,rf_model)
    save_score(score)

