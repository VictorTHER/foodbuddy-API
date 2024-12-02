import pandas as pd
import gcsfs
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from foodbuddy.KNN.KNN_to_prediction import load_KNN
from foodbuddy.RNN.RNN_to_prediction import load_RNN
from foodbuddy.Label_matcher import 
from foodbuddy.params import *

# Instanciate FastAPI
app = FastAPI()

# Load both models
app.state.knn_model = load_KNN()
app.state.rnn_model = load_RNN()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

reminders available;
GCP_PROJECT
GCP_PROJECT_WAGON
GCP_REGION
BUCKET_NAME

@app.get("/nutrients")
def get_nutrients(
        recipe: str,  # "banana bread"
    ):
    """
    Get nutrients for a given recipe
    """
    nutrients =


    print("\nNutrients: ", nutrients, "\n")
    return {'nutrients': nutrients}

@app.get("/recommended")
def get_nutrients(
        nutrients_remaining: list,  # 10 remaining nutrients list
    ):
    """
    Get prediction from KNN model for a givent list of 10 remaining nutrients
    """

    nutrients_preprocessed = preprocess_features(nutrients_remaining)
    y_pred = app.state.model.predict(nutrients_preprocessed)[0][0]
    # KNN returns 10 recipes!

    print("\n✅ prediction done: ", y_pred, "\n")
    return {'10_recipes': float(y_pred)}


@app.get("/")
def root():
    return{'greeting': 'Hello'}
