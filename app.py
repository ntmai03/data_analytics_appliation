import streamlit as st
from PIL import Image

import numpy as np

# Custom imports
from multipage import MultiPage
from page import introduction
from page import houseprice_analysis
from page import time_series
from page import hotel_recommendation
from page import unsupervised_techniques
from page import credit_risk_analysis
from page import shopping_recommendation
from page import coin_trading

# Config layout
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
display = Image.open('image/common/DataScienceBanner.jpg')
display = np.array(display)
#st.image(display, width=800)
#st.image(display)


col1, col2, col3 = st.columns([2,5,2])

with col1:
	st.write("")

with col2:
	st.image(display)

with col3:
	st.markdown("written by [Mai Nguyen](https://www.linkedin.com/in/ntmai03/)")

# Create an instance of the app
app = MultiPage(col1)

# Add all applications here
app.add_page("Select Application", introduction.app)
app.add_page("01-Regression Application", houseprice_analysis.app)
app.add_page("02-Classification Application", credit_risk_analysis.app)
app.add_page("03-Financial Analysis Application", coin_trading.app)
app.add_page("04-Unsupervised Application", unsupervised_techniques.app)
#app.add_page("04-Time Series Application", time_series.app)
app.add_page("05-NLP Application", hotel_recommendation.app)
app.add_page("06-Recommender System Application", shopping_recommendation.app)

# The main app
app.run()





