import streamlit as st
from homepage import homepage
from nutrition_form import nutrition_form
from meal_analysis import meal_analysis

# Access the variable from st.secrets
SERVICE_URL = st.secrets["general"]["SERVICE_URL"]

# Set up session state
if "page" not in st.session_state:
    st.session_state["page"] = "homepage"

# Navigation logic
if st.session_state["page"] == "homepage":
    homepage()
elif st.session_state["page"] == "nutrition_form":
    nutrition_form()
elif st.session_state["page"] == "meal_analysis":
    meal_analysis()
