#Package import
import pandas as pd
import numpy as np
# np.set_printoptions(legacy='1.25') # Making sure float and integers won't show as 'np.float(64)', etc.
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

from foodbuddy.KNN.KNN_model import weighting_nutrients, KNN_model
model_path = "foodbuddy/KNN/fitted_model.pkl"
scaler_path = "foodbuddy/KNN/fitted_scaler.pkl"


def load_model():
    """Load fitted KNN model"""
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            print("KNN Model loaded successfully.")
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model from '{model_path}': {e}")
    else:
        raise FileNotFoundError(f"Model file not found: '{model_path}'")


def load_scaler():
    """Load fitted KNN scaler from a pickle file."""
    if os.path.exists(scaler_path):
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            print("KNN Scaler loaded successfully.")
            return scaler
        except Exception as e:
            raise RuntimeError(f"Error loading scaler from '{scaler_path}': {e}")
    else:
        raise FileNotFoundError(f"Scaler file not found: '{scaler_path}'")


def load_KNN():

    # Check if the pickle file exists
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("Pickle file not found. Generating a new KNN model and scaler...")
        KNN_model()

    # Load the model and scaler
    loaded_model = load_model()
    loaded_scaler = load_scaler()

    return loaded_model, loaded_scaler


def load_user_data(scaler:StandardScaler, weights=None, recommended_intake=None,consumed_intake=None):
    """
    WIP : Dummy user data is being used
    => Replace with user function

    1. Load user consumed and current intake
    2. Calculate the remaining nutrient intake to match the recommended intakes
    """

    # Dummy recommended data for testing
    if recommended_intake is None:
        recommended_intake=np.array([[303., 121.,   81., 560., 4.,  224.,   840., 504., 50., 8.]])

    # Initializing consumed_intakes
    # consumed_intake = np.zeros(len(recommended_intake))
    # Would increment each recipe intake depending on the moment of the day

    # Dummy consumed data for testing
    if consumed_intake is None:
        consumed_intake=np.array([[151.,  60.,  40., 280.,   2., 112., 420., 252., 25.,   4.]])

    # Calculate the remaining intake
    X_remaining=recommended_intake-consumed_intake

    # Scaling the nutrients
    X_remaining_scaled=scaler.transform(X_remaining)

    # Weighting the nutrients
    """WIP To do later : Weight should be customized at some point over each user's importance of reaching certains nutrients"""
    # weights= # Put the user adapted weight algorithm below and before calling weighting_nutrients + Put weights as argument
    X_remaining_weighted=weighting_nutrients(X_remaining_scaled,weights)

    print('Successfully loaded and weighted the user nutritional data')
    return X_remaining_weighted


def predict_KNN_model():
    """MAIN FUNCTION"""
    """Prediction and Nutrition calculation"""
    model=load_model()
    scaler=load_scaler()
    X_remaining_weighted = load_user_data(scaler)

    # Making sure that X is a 2D array:
    if X_remaining_weighted.shape!=(1,10):
        X_remaining_weighted = X_remaining_weighted.reshape(1, -1)

    # Predicting the most nutritious recipes
    y_pred= model.kneighbors(X_remaining_weighted)

    print('Successfully predicted the ideal recipes')

    # Displaying to the terminal the raw KNN output
    print("Here are the raw prediction outputs (recipe matching 'distances', and recipe index in the recipes dataset) :",y_pred)

    # Displaying to the terminal the selected recipes
    # (UTD 12/02/2024 - before GitHub push to main) Making a Python object for Streamlit & API pipeline with the recipe names
    """First step : Loading the recipe names from the model's trained targets, as they contain both recipes indexes and titles"""
    y=pd.read_csv('./recipe_titles.csv')

    print("Here are the selected recipes by order of matching :")
    recommended_recipes_names=[]
    predicted_recipes=y_pred[1][0]
    for i,recipe_index in enumerate(predicted_recipes) :
        print(f"""The recommended recipe n°{i+1} is : {y.loc[recipe_index]['recipe']}.""") # Printing by matching order the selected recipe
        recommended_recipes_names.append(y.loc[recipe_index]['recipe']) # Generating the list of recipe names by matching order for later use
    return y_pred, recommended_recipes_names
    # return y_pred


load_KNN()
