import streamlit as st
from PIL import Image

import numpy as np

# Custom imports
from multipage import MultiPage
from page import introduction
from page import houseprice_analysis


# Config layout
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
display = Image.open('image/common/Logo.png')
display = np.array(display)
col1, col2 = st.columns(2)
col1.image(display, width = 400)
col2.title('Data Analytics Application')

# Create an instance of the app
app = MultiPage()

# Add all applications here
app.add_page("Select Application", introduction.app)
app.add_page("01-House Price Analysis", houseprice_analysis.app)

# The main app
app.run()



