import streamlit as st
import pandas as pd
import requests
from PIL import Image
import toml
import time
import random

# Access the variable from st.secrets
SERVICE_URL = st.secrets["general"]["SERVICE_URL"]
st.write("My Variable:", SERVICE_URL)

### FUNCTIONS AND VARIABLES ###

def call_calculate_api(user_inputs):
    """
    Call the FastAPI /calculate-daily-needs endpoint with user inputs.

    Args:
        user_inputs (dict): Dictionary with user input values.

    Returns:
        dict: API response with BMR, daily caloric needs, and nutrients DataFrame.
    """
    api_url = f"{SERVICE_URL}/calculate-daily-needs"
    response = requests.post(api_url, json=user_inputs)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch daily needs. Status code: {response.status_code}")
        return None

# Add a function to call the FastAPI endpoint for combined analysis
def call_combined_api(image, df):
    """
    Calls the FastAPI /analyze-combined endpoint with the uploaded image and DataFrame.

    Args:
        image (BytesIO): The image file.
        df (pd.DataFrame): The DataFrame with nutrient details.

    Returns:
        dict: API response with analysis results.
    """
    api_url = f"{SERVICE_URL}/analyze-combined"
    files = {"file": image}
    data = {"nutrients": df.to_dict()}  # Serialize DataFrame as dictionary
    response = requests.post(api_url, files=files, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Failed to process data. Status code: {response.status_code}"}


### TITLE AND DESCRIPTION ###

st.markdown(
    """
    <h1 style='text-align: center;'>
    FoodBuddy™ Calculator BETA
    </h1>
    """,
    unsafe_allow_html=True,
)
st.text("Welcome to FoodBuddy™! Our unique model can analize your meal and let you know its nutritional intake.")
st.text(f"Service URL is: {SERVICE_URL}")


# Food picture
st.image("website/HealthyMeal.jpg", caption="Healthy Meal!")

### STEP 1: USER DETAILS FORM ###
st.header("Step 1: Personal Details")

# Collect user inputs
age = st.number_input("Age (years)", min_value=1, max_value=120, step=1)
gender = st.radio("Gender", ["Male", "Female"])
weight = st.number_input("Weight (kg)", min_value=1, max_value=200, step=1)
height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)

st.header("Activity Level")
activity_level = st.selectbox(
    "Choose your activity level",
    [
        "Sedentary (little or no exercise)",
        "Lightly active (light exercise/sports 1-3 days/week)",
        "Moderately active (moderate exercise/sports 3-5 days/week)",
        "Very active (hard exercise/sports 6-7 days a week)",
        "Super active (very hard exercise/physical job)",
    ],
)

# Streamlit Interface
if st.button("Calculate your daily needs!"):
    user_inputs = {
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "activity_level": activity_level,
    }
    with st.spinner("Calculating..."):
        result = call_calculate_api(user_inputs)
        if result:
            bmr = result["bmr"]
            daily_caloric_needs = result["daily_caloric_needs"]
            df = pd.DataFrame(result["nutrients"])

            st.subheader("Your Daily Nutritional Intake")
            st.write(f"**Base Metabolic Rate (BMR):** {bmr} kcal/day")
            # st.write(f"**Total Daily Caloric Needs:** {daily_caloric_needs} kcal/day")
            st.dataframe(df[["Nutrient", "Your Daily Intake", "Description"]])

            st.session_state["df"] = df
            st.session_state["daily_needs_ok"] = True


### STEP 2: PHOTO UPLOAD ###
st.header("Step 2: Scan your meal")

# Split into two columns
col1, col2 = st.columns([1, 1])

with col1:
    # Create a placeholder for the upload button
    upload_placeholder = st.empty()
    uploaded_file = st.file_uploader("Upload a photo of your meal", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        if st.session_state.get("daily_needs_ok", False):
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            with st.spinner("Analyzing your image..."):
                result = call_combined_api(uploaded_file, st.session_state["df"])
                st.session_state["RNN_result"] = result

            if "error" in result:
                st.error(result["error"])
            else:
                st.success("Analysis complete!")
        else :
            st.error("Please fill in personal info to get a personalized nutrients analysis.")

with col2:
    st.text("Processing Status")
    if uploaded_file and st.session_state.get("daily_needs_ok", False):
        # Placeholder for cumulative updates
        status_placeholder = st.empty()

        # List to store messages
        status_messages = []

        # Simulate processing steps with random delays
        steps = [
            "Analyzing your meal...",
            "Identifying ingredients...",
            "Checking nutritional value...",
            "Recommending recipes...",
            "Finalizing results...",
        ]

        for step in steps:
            # Simulate a random delay
            time.sleep(random.uniform(0, 5))

            # Add the new step to the list of messages
            status_messages.append(f"✅ {step}")

            # Add line breaks and update placeholder
            formatted_messages = "<br>".join(status_messages)
            status_placeholder.markdown(
                f"<div style='line-height: 1.8;'>{formatted_messages}</div>",
                unsafe_allow_html=True,
            )

        # Final success message
        st.json(result)


### STEP 3: KNN ###
st.header("Step 3: Find nearest recipes")

# Ensure Step 1 has been completed
if "df" not in st.session_state:
    st.error("Please complete Step 1 to calculate your daily needs first.")
else:
    if st.button("Find Recipes"):
        # Retrieve the user's daily nutrient needs
        user_nutrient_values = st.session_state["df"]["Value"].tolist()

        # Placeholder: Replace with actual RNN output
        # rnn_output = st.session_state.get("RNN_result", [20, 15, 30, 40, 25, 10, 5, 15, 10, 10])  # Mock RNN output
        rnn_output = [20, 15, 30, 40, 25, 10, 5, 15, 10, 10]

        # Compute the difference (remaining nutrients needed)
        nutrient_difference = [max(0, user - rnn) for user, rnn in zip(user_nutrient_values, rnn_output)]

        # Display debug information
        st.subheader("Debug Information")
        st.write("**User Nutrient Values:**", user_nutrient_values)
        st.write("**RNN Output (Predicted Nutrients):**", rnn_output)
        st.write("**Nutrient Difference (Remaining Needs):**", nutrient_difference)

        # Call the KNN API
        api_url = SERVICE_URL
        api_url = f"{SERVICE_URL}/knn-recipes"
        payload = {"nutrient_values": nutrient_difference}
        response = requests.post(api_url, json=payload)

        # Display the results
        if response.status_code == 200:
            knn_results = response.json()

            st.subheader("Recommended Recipes")
            for recipe in knn_results["recipes"]:
                st.markdown(f"- **{recipe['recipe']}** (Distance: {recipe['distance']})")
        else:
            st.error(f"Failed to fetch recipes. Status code: {response.status_code}")
