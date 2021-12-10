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
from analysis.hotel_recommendation import HotelRecommendation


def app():

    st.sidebar.header('')
    st.sidebar.header('')
    st.sidebar.header('')
    st.sidebar.subheader('Select function')
    task_type = ["Extract Data", 
                 "Transform Data",
                 "Data Processing",
                 "Text Classification",
                 "Topic Modeling",
                 "Hotel Recommendation",
                 "Knowledge Graph"]
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
            Booking.create_review_file()
        if(location_option == 'Paris'):
            Booking = BookingScrapper()
            Booking.search_accommodation(dest_id='-1456928', city='Paris')
            Booking.get_review()
            Booking.create_review_file()

    if task_option == 'Transform Data':
        location_list = ['Select city...',
                         'London',
                         'Paris',
                         'New York']
        location_option = st.sidebar.selectbox('',location_list)
        location_dict = {'London':'-2601889', 'Paris':'-1456928'}

        if(location_option == 'London'):
            Booking = BookingScrapper()
            Booking.create_review_file(location_option)
        if(location_option == 'Paris'):
            Booking = BookingScrapper()
            Booking.create_review_file(location_option)


    if task_option == 'Topic Modeling':
        st.sidebar.header('')
        st.sidebar.header('')
        st.sidebar.subheader('Dimension Reduction')
        source_type = ["Select Data Source",
                          "AirBnb", 
                          "News"]
        source_option = st.sidebar.selectbox('',source_type)
        st.sidebar.header('')
        st.sidebar.header('')


    if task_option == 'Knowledge Graph':
        st.sidebar.header('')
        st.sidebar.header('')
        
        LISTING_ID = st.sidebar.number_input(
            "Input listing_id",   
            min_value=1296836,
            key="LISTING_ID",
        )

        if "ENTITY_OPTION" not in st.session_state:
            st.session_state.ENTITY_OPTION = []
        if "ALL_ENTITIES" not in st.session_state:
            st.session_state.ALL_ENTITIES = []

        st.sidebar.markdown('#### Select cluster')
        cluster_type = ["0", 
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7"]
        cluster_option = st.sidebar.selectbox('',cluster_type)
        st.sidebar.header('')

        ENTITY_OPTION = st.multiselect("Select entities", options=st.session_state.ALL_ENTITIES, default=st.session_state.ENTITY_OPTION)

        if st.sidebar.button('Generate Graph'):

            if(len(st.session_state.ENTITY_OPTION) > 0):
                st.session_state.ENTITY_OPTION = ENTITY_OPTION

            hotel_recommendation = HotelRecommendation()
            entities, flag = hotel_recommendation.generate_graph(listing_id = int(LISTING_ID),cluster_id=int(cluster_option), entities=st.session_state.ENTITY_OPTION)                  
            if(flag == 1):
                st.session_state.ALL_ENTITIES = entities
            st.session_state.ENTITY_OPTION = entities
            ENTITY_OPTION = st.multiselect("Select entities", options=st.session_state.ALL_ENTITIES, default=st.session_state.ENTITY_OPTION)
        

        
        
        
            



    
