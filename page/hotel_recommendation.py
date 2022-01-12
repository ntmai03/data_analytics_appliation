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
    task_type = ["Select function",
                 "Extract Data", 
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
            Booking.search_accommodation(dest_id='-2601889', city='london')
            Booking.get_review()
        if(location_option == 'Paris'):
            Booking = BookingScrapper()
            Booking.search_accommodation(dest_id='-1456928', city='paris')
            Booking.get_review()

    if task_option == 'Transform Data':
        location_list = ['Select city...',
                         'London',
                         'Paris',
                         'New York']
        location_option = st.sidebar.selectbox('',location_list)
        location_dict = {'London':'-2601889', 'Paris':'-1456928'}

        if(location_option == 'London'):
            Booking = BookingScrapper()
            Booking.create_review_file(city='london')
        if(location_option == 'Paris'):
            Booking = BookingScrapper()
            Booking.create_review_file(city='paris')



    if task_option == 'Data Processing':
        source_type = ["Select Data Source",
                          "London", 
                          "Paris"]
        source_option = st.sidebar.selectbox('',source_type)
        source_dict = {'London':'london', 'Paris':'paris'}
        st.sidebar.header('')
        st.sidebar.header('')
        if(source_option != 'Select Data Source'):
            hotel = HotelRecommendation()
            hotel.create_corpus(source_dict[source_option])


    if task_option == 'Topic Modeling':
        source_type = ["Select Data Source",
                          "AirBnb", 
                          "London",
                          "Paris"]
        source_option = st.sidebar.selectbox('',source_type)
        source_dict = {'London':'london', 'Paris':'paris'}
        st.write()
        st.sidebar.markdown('N of clusters')
        n_clusters = st.sidebar.slider("",2, 10, 7, key="N_CLUSTERS")
        st.sidebar.markdown('N of hidden nodes in the 1st layer')
        encoding1_dim = st.sidebar.slider("",50, 100, 80, key="ENCODING1_DIM")
        st.sidebar.markdown('N of hidden nodes in the 2nd layer')
        encoding2_dim = st.sidebar.slider("",20, 50, 30, key="ENCODING2_DIM")
        st.sidebar.markdown('N of hidden nodes in the 2nd layer')
        latent_dim = st.sidebar.slider("",10, 20, 15, key="LATENT_DIM")
        st.sidebar.header('')

        if st.sidebar.button("Train"):
            if(source_option != 'Select Data Source'):
                hotel = HotelRecommendation()
                hotel.train_autoencoder(source_dict[source_option], 
                                        encoding1_dim, 
                                        encoding2_dim, 
                                        latent_dim, 
                                        n_clusters)


    if task_option == 'Hotel Recommendation':
        st.write('HotelRecommendation')
        hotel = HotelRecommendation()
        hotel.hotel_recommendation()


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
        

        
        
        
            



    
