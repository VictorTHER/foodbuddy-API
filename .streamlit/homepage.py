import streamlit as st

# Access the variable from st.secrets
SERVICE_URL = st.secrets["general"]["SERVICE_URL"]

def homepage():
    st.title("🍴 Welcome to FoodBuddy!")
    st.image(".streamlit/HealthyMeal.jpg", caption="Healthy Meal!")
    st.markdown(
        """
        FoodBuddy™ helps you analyze your meal and provides nutritional insights.
        Click **Next** to get started.
        """
    )

    if st.button("Next"):
        st.session_state["page"] = "nutrition_form"
        st.experimental_rerun()
