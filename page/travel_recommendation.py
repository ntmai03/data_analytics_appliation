import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
import os

sys.path.append('src')
from src import config as cf
from src.util import data_manager as dm
from src.data_processing import diabetes_feature_engineering as fe
from util.booking_scrapper import BookingScrapper


def app():

    st.sidebar.subheader('Select function')
    task_type = ["Extract Data", 
                 "Reference",
                 "Data Processing",
                 "Predictive Model",
                 "Prediction"]
    task_option = st.sidebar.selectbox('',task_type)
    st.sidebar.header('')

    if task_option == 'Extract Data':
        location_list = ['Select city...',
                         'London',
                         'Paris',
                         'New York']
        location_option = st.sidebar.selectbox('',location_list)
        location_dict = {'London':'-2601889', 'Paris':'-1456928'}

        if(location_option == 'London'):
            Booking = BookingScrapper()
            Booking.search_accommodation(dest_id='-2601889', city='London')
            Booking.get_review()
        if(location_option == 'Paris'):
            Booking = BookingScrapper()
            Booking.search_accommodation(dest_id='-1456928', city='Paris')
            Booking.get_review()