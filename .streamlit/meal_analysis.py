import streamlit as st
import requests
import pandas as pd
import random
import time

# Access the variable from st.secrets
SERVICE_URL = st.secrets["general"]["SERVICE_URL"]


def remaining_nutrients_manual(df, detected_recipe_df):
    """
    Manually aligns and matches nutrient columns between the user's daily intake
    and detected recipe content. Calculates the remaining daily intake.

    Args:
        df (pd.DataFrame): User's daily intake with columns ["Nutrient", "Your Daily Intake"].
        detected_recipe_df (pd.DataFrame): Detected recipe content with columns ["Nutrient", "Value"].

    Returns:
        pd.DataFrame: A DataFrame showing remaining daily intake, original intake, and detected values.
    """

    # Define manual column mapping
    nutrient_mapping = {
        "Carbohydrates": "Carbohydrates_(G)_total",
        "Proteins": "Protein_(G)_total",
        "Fats": "Lipid_(G)_total",
        "Calcium": "Calcium_(MG)_total",
        "Iron": "Iron_(MG)_total",
        "Magnesium": "Magnesium_(MG)_total",
        "Sodium": "Sodium_(MG)_total",
        "Vitamin C": "Vitamin_C_(MG)_total",
        "Vitamin D": "Vitamin_D_(UG)_total",
        "Vitamin A": "Vitamin_A_(UG)_total",
    }

    # Prepare daily intake values (strip units and convert to floats)
    df["Daily Intake (Value)"] = df["Your Daily Intake"].str.extract(r"([\d\.]+)").astype(float)

    # Align detected recipe values using the mapping
    detected_values = detected_recipe_df.loc[list(nutrient_mapping.values()), 0]

    # Perform subtraction to calculate remaining nutrients
    remaining_nutrients = df["Daily Intake (Value)"].values - detected_values.values

    # Create output DataFrame
    remaining_df = pd.DataFrame({
        "Nutrient": df["Nutrient"],
        "Remaining Daily Intake": remaining_nutrients,
        "Original Daily Intake": df["Daily Intake (Value)"].values,
        "Detected Plate Content": detected_values.values,
    })

    return remaining_df


def meal_analysis():

    # Add a "Go Back" button to navigate to the previous page
    if st.button("Go Back"):
        st.session_state["page"] = "nutrition_form"
        st.experimental_rerun()

    st.title("Analyze Your Meal")
    st.markdown("Upload an image of your meal to analyze its nutritional content.")

    # Upload Section
    uploaded_file = st.file_uploader("Upload a photo of your meal", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        if st.button("Analyze Plate Content"):
            with st.spinner("Analyzing your image..."):
                time.sleep(random.uniform(1, 3))  # Simulate processing delay

                try:
                    # Send the file to the API for plate analysis
                    files = {"file": uploaded_file.getvalue()}
                    api_url = f"{SERVICE_URL}/analyze-image"
                    response = requests.post(api_url, files=files)

                    if response.status_code == 200:
                        result = response.json()

                        if "error" in result:
                            st.error(f"Error: {result['error']}")
                        else:
                            confidence = result["probability"]
                            recipe_name = result["predicted_recipe_name"]
                            if confidence > 0.8:
                                st.success(f"We detected **{recipe_name}** on your plate with high confidence!")

                                # Fetch nutrients for the detected recipe
                                nutrients_url = f"{SERVICE_URL}/tnutrients?recipe={recipe_name}"
                                nutrients_response = requests.get(nutrients_url)

                                if nutrients_response.status_code == 200:
                                    nutrients = nutrients_response.json().get("nutrients", [])
                                    detected_recipe_df = pd.DataFrame(nutrients).filter(regex="total$").transpose()

                                    st.subheader("Nutritional Information")
                                    st.dataframe(detected_recipe_df)

                                    remaining_df = remaining_nutrients_manual(st.session_state.get("df"),detected_recipe_df)

                                    # Automatically call KNN after plate analysis
                                    nutrient_values = remaining_df["Remaining Daily Intake"].tolist()
                                    if nutrient_values is not None:
                                        # Prepare payload for KNN
                                        payload = {
                                            "nutrient_values": nutrient_values,
                                        }

                                        # Call KNN API
                                        knn_url = f"{SERVICE_URL}/knn-recipes"
                                        knn_response = requests.post(knn_url, json=payload)

                                        if knn_response.status_code == 200:
                                            knn_results = knn_response.json()
                                            st.subheader("Recommended Recipes")
                                            for recipe in knn_results.get("recipes", []):
                                                st.markdown(f"- **{recipe['recipe']}**")
                                        else:
                                            st.error(f"KNN API call failed with status code {knn_response.status_code}")
                                    else:
                                        st.error("User's daily nutrient data not found.")

                            elif 0.6 <= confidence <= 0.8:
                                st.warning(f"Your meal might be **{recipe_name}**. The model has moderate confidence.")

                                # Prompt user feedback
                                feedback_col1, feedback_col2 = st.columns(2)
                                with feedback_col1:
                                    if st.button("Yes, that's correct!"):
                                        nutrients_url = f"{SERVICE_URL}/tnutrients?recipe={recipe_name}"
                                        nutrients_response = requests.get(nutrients_url)
                                        if nutrients_response.status_code == 200:
                                            nutrients = nutrients_response.json().get("nutrients", [])
                                            detected_recipe_df = pd.DataFrame(nutrients).filter(regex="total$").transpose()
                                            st.subheader("Nutritional Information")
                                            st.dataframe(detected_recipe_df)

                                            # Call KNN as above
                                            user_daily_nutrients = st.session_state.get("df")
                                            if user_daily_nutrients is not None:
                                                # Prepare payload for KNN
                                                payload = {
                                                    "user_daily_nutrients": user_daily_nutrients.to_dict(orient="records"),
                                                    "detected_recipe_nutrients": detected_recipe_df.to_dict(orient="records"),
                                                }

                                                # Call KNN API
                                                knn_url = f"{SERVICE_URL}/knn-recipes"
                                                knn_response = requests.post(knn_url, json=payload)

                                                if knn_response.status_code == 200:
                                                    knn_results = knn_response.json()
                                                    st.subheader("Recommended Recipes")
                                                    for recipe in knn_results.get("recipes", []):
                                                        st.markdown(f"- **{recipe['recipe']}** (Distance: {recipe['distance']:.2f})")
                                                else:
                                                    st.error(f"KNN API call failed with status code {knn_response.status_code}")
                                            else:
                                                st.error("User's daily nutrient data not found.")
                                        else:
                                            st.error("Unable to fetch nutrients for the identified recipe.")

                                with feedback_col2:
                                    if st.button("No, that's not correct."):
                                        st.info("Please upload a new photo for analysis.")

                            elif 0.4 <= confidence < 0.6:
                                st.warning("The model is unsure about your meal. Could you help us improve?")
                                user_input = st.text_input("What is on your plate?")
                                if user_input:
                                    st.success("Thank you for helping us improve! Please upload another photo if needed.")

                            else:
                                st.error("We couldn't confidently identify your meal. Please try again!")


                    else:
                        st.error(f"API call failed with status code {response.status_code}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
