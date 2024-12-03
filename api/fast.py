### STEP 1: IMPORT PACKAGES, VARIABLES AND FUNCTIONS ###

# Import packages
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import functions
from foodbuddy.KNN.KNN_to_predictions import load_KNN, weighting_nutrients
from foodbuddy.RNN.RNN_to_prediction import load_RNN
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

# lifespan = set up downloads of data+models when turning API on!
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
    app.state.knn_model, app.state.knn_scaler = load_KNN()
    app.state.rnn_model = load_RNN()

    # Marks API running OK point
    yield

    # Shutdown: Cleanup things!
    del app.state.targets
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
    Upload .jpeg photo
    Analyze with model
    Output result of model analysis
    """
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

    try:
        # Read the image file as bytes
        image_bytes = await file.read()

        # Call the analyze_image function
        result = analyze_image(image_bytes)

        return result
    except Exception as e:
        return {"error": f"Image analysis failed: {str(e)}"}


# KNN predict 10 recipes for a given filled form
@app.post("/knn-recipes")
def knn_recipes(nutrient_values: list):
    """
    Input a list of 10 nutrient values
    Output a JSON dict with 10 nearest recipes and details

    """
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

    # Sdddames
    knn_model = app.state.knn_model
    knn_scaler = app.state.knn_scaler

    nutrient_values_scaled = knn_scaler.fit_transform(nutrient_values)
    nutrient_values_weighted = weighting_nutrients(nutrient_values_scaled)

    distances, indices = knn_model.kneighbors([nutrient_values_weighted], n_neighbors=10)
    recipe_names = app.state.recipes.iloc[indices[0]]["recipe"].tolist()

    # Step 3: Query the recipes database using the names
    queried_recipes = query_recipes(recipe_names)

    # Step 4: Add distances to the queried recipes for Streamlit
    for i, recipe in enumerate(queried_recipes["recipes"]):
        recipe["distance"] = round(distances[0][i], 4)  # Include distance as a field

    return queried_recipes



# Get nutrients for a given recipe
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


# API health checker
@app.get("/")
def root():
    """
    Test endpoint to verify the API is running.
    """
    return {"message": "API is up and running :) "}
