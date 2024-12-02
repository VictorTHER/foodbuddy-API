import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

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
