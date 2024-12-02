import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
import io


from foodbuddy.KNN.KNN_to_predictions import load_KNN
from foodbuddy.RNN.RNN_to_prediction import load_RNN
from foodbuddy.Label_matcher.Target_match_setup import download_targets_df
from foodbuddy.Label_matcher.Recipes_list_setup import download_recipes_df
from foodbuddy.Label_matcher.Ingredients_list_setup import download_ingredients_df
from foodbuddy.params import *


## STEP 1: SET UP DF DOWNLOADS ###

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

    # Marks API running OK point
    yield

    # Shutdown: Cleanup things!
    del app.state.targets
    del app.state.recipes
    del app.state.ingredients
    del app.state.knn_model
    del app.state.rnn_model


## STEP 2: INSTANCIATE API AND PULL MODELS ###

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
# TEST: http://127.0.0.1:8000/nutrients?recipe=banana%20bread
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


def query_recipes(recipe_names: list):
    """
    Query the recipes database using a list of recipe names.

    Args:
        recipe_names (list): List of recipe names.

    Returns:
        dict: A dictionary containing the matched recipes.
    """
    # Get the recipes DataFrame from the API state
    recipes_df = app.state.recipes

    # Query the DataFrame using the recipe names
    filtered_recipes = recipes_df[recipes_df["recipe"].isin(recipe_names)]

    # Convert the filtered DataFrame to a JSON-compatible format
    results = filtered_recipes.to_dict(orient="records")

    return {"recipes": results}


@app.post("/knn-recipes")
def knn_recipes(nutrient_values: list):
    """
    Input a list of 10 nutrient values
    Output a JSON dict with 10 nearest recipes and details

    """
    # Step 1: Use KNN model to get 10 nearest recipe names
    knn_model = app.state.knn_model
    distances, indices = knn_model.kneighbors([nutrient_values], n_neighbors=10)
    recipe_names = app.state.recipes.iloc[indices[0]]["recipe"].tolist()

    # Step 2: Query the recipes database using the names
    queried_recipes = query_recipes(recipe_names)

    # Step 3: Add distances to the queried recipes for Streamlit
    for i, recipe in enumerate(queried_recipes["recipes"]):
        recipe["distance"] = round(distances[0][i], 4)  # Include distance as a field

    return queried_recipes


def analyze_image(image: bytes):
    """
    Input .jpg image
    Analyze an image using the RNN model.
    Return the result of the RNN model analysis.
    """
    # Step 1: Ensure the RNN model is loaded
    rnn_model = app.state.rnn_model

    # Step 2: Run the image through the RNN model
    try:
        prediction = rnn_model.predict(image)
    except Exception as e:
        return {"error": f"Model prediction failed: {str(e)}"}

    # Step 3: Format and return the output
    return {"prediction": prediction.tolist()}


@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Upload .jpeg photo
    Analyze with model
    Output result of model analysis
    """
    try:
        # Read the image file as bytes
        image_bytes = await file.read()

        # Call the analyze_image function
        result = analyze_image(image_bytes)

        return result
    except Exception as e:
        return {"error": f"Image analysis failed: {str(e)}"}

@app.get("/")
def root():
    """
    Test endpoint to verify the API is running.
    """
    return {"message": "Welcome to the Nutrients API"}
