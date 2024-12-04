import streamlit as st
import requests
import pandas as pd

# Access the variable from st.secrets
SERVICE_URL = st.secrets["general"]["SERVICE_URL"]

def meal_analysis():
    st.title("Step 2: Analyze Your Meal")
    st.markdown("Upload an image of your meal to analyze its nutritional content.")

    # Split into two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        # Create a placeholder for the upload button
        uploaded_file = st.file_uploader("Upload a photo of your meal", type=["jpg", "png", "jpeg"])

        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.subheader("Processing Status")

        if uploaded_file:
            # Proceed with image analysis
            if st.button("What Should I Eat Tonight?"):
                with st.spinner("Analyzing your image..."):
                    try:
                        # Send the file to the API
                        files = {"file": uploaded_file.getvalue()}
                        api_url = f"{SERVICE_URL}/analyze-image"
                        response = requests.post(api_url, files=files)

                        if response.status_code == 200:
                            result = response.json()

                            # Handle the response based on confidence
                            if "error" in result:
                                st.error(f"Error: {result['error']}")
                            else:
                                confidence = result["probability"]
                                recipe_name = result["predicted_recipe_name"]

                                if confidence > 0.8:
                                    st.success(f"We detected **{recipe_name}** on your plate!")
                                    # Optionally fetch recipe nutrients
                                    nutrients_url = f"{SERVICE_URL}/nutrients?recipe={recipe_name}"
                                    nutrients_response = requests.get(nutrients_url)
                                    if nutrients_response.status_code == 200:
                                        nutrients = nutrients_response.json().get("nutrients", [])
                                        df = pd.DataFrame(nutrients).filter(regex="total$")
                                        st.dataframe(df)
                                elif 0.6 <= confidence <= 0.8:
                                    st.warning(f"Your meal might be **{recipe_name}**. Are we correct?")
                                    if st.button("Yes"):
                                        nutrients_url = f"{SERVICE_URL}/nutrients?recipe={recipe_name}"
                                        nutrients_response = requests.get(nutrients_url)
                                        if nutrients_response.status_code == 200:
                                            nutrients = nutrients_response.json().get("nutrients", [])
                                            df = pd.DataFrame(nutrients).filter(regex="total$")
                                            st.dataframe(df)
                                    elif st.button("No"):
                                        st.info("Please upload a new photo.")
                                else:
                                    st.error("We couldn't confidently identify your meal. Please try again!")
                        else:
                            st.error(f"API call failed with status code {response.status_code}")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    # Add a "Go Back" button to navigate to the previous page
    if st.button("Go Back"):
        st.session_state["page"] = "nutrition_form"
