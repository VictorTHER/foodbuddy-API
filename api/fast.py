### STEP 1: IMPORT PACKAGES, VARIABLES AND FUNCTIONS ###

# Import packages
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import os
from io import BytesIO
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import functions
from foodbuddy.KNN.KNN_to_predictions import load_KNN, weighting_nutrients
from foodbuddy.RNN.load_RNN import load_RNN
from foodbuddy.RNN.Image_preproc import full_pipeline
from foodbuddy.Label_matcher.Target_match_setup import download_targets_df
from foodbuddy.Label_matcher.Recipes_list_setup import download_recipes_df
from foodbuddy.Label_matcher.Ingredients_list_setup import download_ingredients_df

# Import parameters (API URL, etc.)
from foodbuddy.params import *



### STEP 2: SETUP API ###

# Prepare "UserInputs" class to handle daily needs calculation
class UserInputs(BaseModel):
    age: int
    gender: str
    weight: float
    height: float
    activity_level: str

# Prepare "NutrientValues" class to handle KNN calculations
class NutrientValues(BaseModel):
    nutrient_values: List[float]  # Ensure a list of floats is expected

# lifespan = set up downloads of data+models when turning API on!
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context for loading resources on startup and cleanup on shutdown.
    """
    # Load data
    targets_df = download_targets_df()
    app.state.targets = targets_df
    app.state.target = targets_df["recipe"].tolist()
    app.state.nutrients = targets_df.drop(columns=["recipe"])

    app.state.recipes = download_recipes_df()
    app.state.ingredients = download_ingredients_df()

    # Load models
    app.state.knn_model, app.state.knn_scaler = load_KNN()
    app.state.rnn_model = load_RNN()

    # Marks API running OK point
    yield

    # Shutdown: Cleanup things!
    del app.state.target
    del app.state.targets
    del app.state.nutrients
    del app.state.recipes
    del app.state.ingredients
    del app.state.knn_model
    del app.state.rnn_model



## STEP 3: INSTANCIATE API ###

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


### STEP 4: MAKE ENDPOINTS! ###

# Calculate user daily needs df using form
@app.post("/calculate-daily-needs")
def calculate_daily_needs(user: UserInputs):
    activity_multipliers = {
        "Sedentary (little or no exercise)": 1.2,
        "Lightly active (light exercise/sports 1-3 days/week)": 1.375,
        "Moderately active (moderate exercise/sports 3-5 days/week)": 1.55,
        "Very active (hard exercise/sports 6-7 days a week)": 1.725,
        "Super active (very hard exercise/physical job)": 1.9,
    }

    # Validate gender
    if user.gender not in ["Male", "Female"]:
        raise HTTPException(status_code=400, detail="Invalid gender value")

    # Calculate BMR
    if user.gender == "Male":
        bmr = 88.362 + (13.397 * user.weight) + (4.799 * user.height) - (5.677 * user.age)
    else:
        bmr = 447.593 + (9.247 * user.weight) + (3.098 * user.height) - (4.330 * user.age)

    # Adjust for activity level
    activity_multiplier = activity_multipliers.get(user.activity_level)
    if not activity_multiplier:
        raise HTTPException(status_code=400, detail="Invalid activity level")

    daily_caloric_needs = bmr * activity_multiplier

    # Macronutrient calculations in grams
    macros = {
        "Carbohydrates": (daily_caloric_needs * 0.5) / 4,
        "Proteins": (daily_caloric_needs * 0.2) / 4,
        "Fats": (daily_caloric_needs * 0.3) / 9,
    }

    # Micronutrient calculations in respective units
    micronutrients = {
        "Calcium": 1000 * (bmr / 2796),  # mg
        "Iron": 8 * (bmr / 2796),        # mg
        "Magnesium": 400 * (bmr / 2796), # mg
        "Sodium": 1500 * (bmr / 2796),   # mg
        "Vitamin C": 90 * (bmr / 2796),  # mg
        "Vitamin D": 15 * (bmr / 2796),  # µg
        "Vitamin A": 900 * (bmr / 2796), # µg
    }

    # Combine results
    nutrients = {**macros, **micronutrients}
    units = ["g"] * 3 + ["mg"] * 5 + ["µg"] * 2
    nutrient_names = list(nutrients.keys())
    daily_intake = list(nutrients.values())
    descriptions = [
        "Primary energy source for body, fuels brain and muscles.",
        "Builds and repairs tissues, supports enzymes and hormone production.",
        "Stores energy, supports cell growth, and protects organs.",
        "Essential for strong bones, teeth, and muscle function.",
        "Vital for oxygen transport in blood and energy production.",
        "Supports nerve function, muscle contraction, and heart health.",
        "Regulates fluid balance, muscle contraction, and nerve signals.",
        "Boosts immune function, repairs tissues, and antioxidant protection.",
        "Helps calcium absorption, maintains bone health and immune function.",
        "Supports vision, skin health, and immune system functionality.",
    ]

    # Create DataFrame
    df = pd.DataFrame({
        "Nutrient": nutrient_names,
        "Value": [round(value) for value in daily_intake],
        "Unit": units,
        "Your Daily Intake": [f"{round(value)} {unit}" for value, unit in zip(daily_intake, units)],
        "Description": descriptions
    })

    # Return BMR, daily needs, and DataFrame as JSON
    return {
        "bmr": round(bmr),
        "daily_caloric_needs": round(daily_caloric_needs),
        "nutrients": df.to_dict(orient="records")
    }


# RNN predict dish
@app.post("/analyze-image")
async def analyze_image_endpoint(file: UploadFile = File(...)):
    """
    Upload .jpg photo
    Analyze with model after preprocessing pipeline
    Output result of model analysis
    """
    try:
        # Save uploaded file as a temporary file
        temp_dir = "temp_images"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file.filename)

        with open(temp_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Run the image through the preprocessing pipeline
        img = Image.open(temp_path)

        # Preprocess: Resize to 224x224, convert to RGB, normalize
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Run the preprocessed image through the RNN model
        rnn_model = app.state.rnn_model
        prediction = rnn_model.predict(img_array)

        # Get the index of the highest probability
        max_index = int(np.argmax(prediction))
        print(max_index)

        # Access the maximum probability value
        max_probability = float(prediction[0, max_index])  # Correct indexing for 2D array
        print(max_probability)

        # Use app.state.targets for mapping
        recipe_name = app.state.target[max_index]

        # Clean up temporary files
        os.remove(temp_path)

        return {
            "predicted_recipe_index": max_index,
            "predicted_recipe_name": recipe_name,
            "probability": max_probability,
        }
    except Exception as e:
        return {"error": f"Image analysis failed: {str(e)}"}


# KNN predict 10 recipes for a given filled form
@app.post("/knn-recipes")
def knn_recipes(payload: NutrientValues):
    """
    Input a list of 10 nutrient values
    Output a JSON dict with 10 nearest recipes and details
    """
    nutrient_values = payload.nutrient_values  # Extract nutrient values from the payload

    def query_recipes(recipe_names: list):
        # Get the recipes DataFrame from the API state
        recipes_df = app.state.recipes
        filtered_recipes = recipes_df[recipes_df["recipe"].isin(recipe_names)]
        results = filtered_recipes.to_dict(orient="records")
        return {"recipes": results}

    # Access KNN model and scaler
    knn_model = app.state.knn_model
    knn_scaler = app.state.knn_scaler

    # Process the input
    nutrient_values_scaled = knn_scaler.transform([nutrient_values])  # Use transform instead of fit_transform

    # Get the nearest neighbors
    distances, indices = knn_model.kneighbors(nutrient_values_scaled, n_neighbors=10)
    recipe_names = app.state.recipes.iloc[indices[0]]["recipe"].tolist()

    # Query recipes and add distances
    queried_recipes = query_recipes(recipe_names)
    for i, recipe in enumerate(queried_recipes["recipes"]):
        recipe["distance"] = round(distances[0][i], 4)

    return queried_recipes


# Get nutrients for a given recipe
# TEST: http://127.0.0.1:8000/nutrients?recipe=banana%20bread
@app.get("/rnutrients")
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

# Get nutrients for a given target
@app.get("/tnutrients")
def get_nutrients(recipe: str):
    """
    Get nutrients for a given target.
    """
    # Get data (already up with API :)
    df = app.state.targets

    # Save to nutrients in a JSON/API compatible format (orient="records")
    nutrients = df[df["recipe"].str.lower() == recipe.lower()].to_dict(orient="records")

    if not nutrients:
        return {"message": "Recipe not found"}

    return {"nutrients": nutrients}

# API health checker
@app.get("/")
def root():
    """
    Test endpoint to verify the API is running.
    """
    return {"message": "API is up and running :) ",
            "target length": len(app.state.target),
            "target look":app.state.target}
