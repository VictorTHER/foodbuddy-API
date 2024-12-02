import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from foodbuddy.KNN.KNN_to_prediction import load_KNN
from foodbuddy.RNN.RNN_to_prediction import load_RNN
from foodbuddy.Label_matcher.Target_match_setup import download_targets_df
from foodbuddy.Label_matcher.Recipes_list_setup import download_recipes_df
from foodbuddy.Label_matcher.Ingredients_list_setup import download_ingredients_df
from foodbuddy.params import *


### STEP 1: SET UP DF DOWNLOADS ###

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context for loading resources on startup and cleanup on shutdown.
    """
    # Load data
    app.state.targets = download_targets_df()
    app.state.recipes = download_recipes_df()
    app.state.ingredients = download_ingredients_df()

    # Load models
    app.state.knn_model = load_KNN()
    app.state.rnn_model = load_RNN()

    # ????
    yield

    # Shutdown: Cleanup things!
    del app.state.targets
    del app.state.recipes
    del app.state.ingredients
    del app.state.knn_model
    del app.state.rnn_model


### STEP 2: INSTANCIATE API AND PULL MODELS ###

# Instanciate FastAPI with lifespan handler
app = FastAPI(lifespan=lifespan)

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


### STEP 3: GET NUTRIENTS WITH A RECIPE NAME ###
@app.get("/nutrients")
def get_nutrients(recipe: str):
    """
    Get nutrients for a given recipe.
    """
    # Get data (already up with API :)
    df = app.state.recipes

    # Save to nutrients in a JSON/API compatible format (orient="records")
    nutrients = df[df["recipe"].str.lower() == recipe.lower()].to_dict(orient="records")

    if not nutrients:
        return {"message": "Recipe not found"}

    return {"nutrients": nutrients}


# @app.get("/recommended")
# def get_nutrients(
#         nutrients_remaining: list,  # 10 remaining nutrients list
#     ):
#     """
#     Get prediction from KNN model for a givent list of 10 remaining nutrients
#     """

#     nutrients_preprocessed = preprocess_features(nutrients_remaining)
#     y_pred = app.state.model.predict(nutrients_preprocessed)[0][0]
#     # KNN returns 10 recipes!

#     print("\n✅ prediction done: ", y_pred, "\n")
#     return {'10_recipes': float(y_pred)}


@app.get("/")
def root():
    """
    Test endpoint to verify the API is running.
    """
    return {"message": "Welcome to the Nutrients API"}
