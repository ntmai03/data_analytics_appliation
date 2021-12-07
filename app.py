import streamlit as st
from PIL import Image

import numpy as np

# Custom imports
from multipage import MultiPage
from page import introduction
from page import houseprice_analysis
from page import time_series
from page import travel_recommendation
from page import unsupervised_techniques
from page import credit_risk_analysis

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

#col1, col2 = st.columns(2)
#col1.image(display, width = 400)
#new_title = '<p style="font-family:sans-serif; color:LightGreen; font-size: 42px;">Data Analytics Application</p>'
#col2.markdown(new_title, unsafe_allow_html=True)


#col2.header('Data Analytics Application')

# Create an instance of the app
app = MultiPage(col1)

# Add all applications here
app.add_page("Select Application", introduction.app)
app.add_page("01-House Price Data Analysis", houseprice_analysis.app)
app.add_page("02-Credit Risk Analysis", credit_risk_analysis.app)
app.add_page("03-Unsupervised Techniques", unsupervised_techniques.app)
app.add_page("04-Time Series Data Analysis", time_series.app)
app.add_page("05-Text Data Analysis", introduction.app)
app.add_page("06-Travel Recommendation", travel_recommendation.app)
app.add_page("07-Online Shopping Recommendation", introduction.app)

# The main app
app.run()





