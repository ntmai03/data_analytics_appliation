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
from data_processing.booking_scrapper import BookingScrapper
from analysis.hotel_recommendation import HotelRecommendation


def app():

    st.sidebar.header('')
    st.sidebar.header('')  
    new_city = st.sidebar.text_input('Input your destination').capitalize()
    if st.sidebar.button('Add'):
        Booking = BookingScrapper()
        cf.data['dest_name'].update({new_city:str.lower(new_city)})
        dest_id, _ = Booking.search_location(str.lower(new_city))
        cf.data['dest_id'].update({new_city:dest_id})
        cf.update_yaml_config_file(cf.data)

    location_list = ['Select city...'] + list(cf.data['dest_name'].keys())
    location_option = st.sidebar.selectbox('',location_list)
    source_dict = cf.data['dest_name']
    location_dict = cf.data['dest_id']
    st.sidebar.header('')
    st.sidebar.header('')


    task_type = ["Select function",
                 "Extract Data", 
                 "Transform Data",
                 "Data Processing",
                 "Topic Modeling",
                 "Hotel Recommendation",
                 "Knowledge Graph",
                 "Text Classification",]
    task_option = st.sidebar.selectbox('',task_type)
    st.sidebar.header('')


    if ((task_option == 'Extract Data') & (location_option != 'Select city...')):
            Booking = BookingScrapper()
            _, nr_hotels = Booking.search_location(source_dict[location_option])
            nr_hotels = int(nr_hotels)
            hotels = st.sidebar.slider("Select number of hotels",20, nr_hotels, value=60, step=20, key="N_HOTELS")
            st.write(hotels)
            if st.sidebar.button("Download"):
                Booking.search_accommodation(dest_id=int(location_dict[location_option]), city=source_dict[location_option], nr_hotels=int(hotels))
                Booking.get_review()


    if ((task_option == 'Transform Data') & (location_option != 'Select city...')):
            Booking = BookingScrapper()
            Booking.create_review_file(city=source_dict[location_option])


    if task_option == 'Data Processing':
        if(location_option != 'Select city...'):
            hotel = HotelRecommendation()
            hotel.text_processing(source_dict[location_option])


    if task_option == 'Topic Modeling':
        if(location_option == 'AirBnb'):
            hotel = HotelRecommendation()
        if((location_option != 'Select city...') & (location_option != 'AirBnb')):
            st.sidebar.markdown('N of clusters')
            n_clusters = st.sidebar.slider("",2, 10, 7, key="N_CLUSTERS")
            st.sidebar.markdown('N of hidden nodes in the 1st layer')
            encoding1_dim = st.sidebar.slider("",50, 300, 100, key="ENCODING1_DIM")
            st.sidebar.markdown('N of hidden nodes in the 2nd layer')
            encoding2_dim = st.sidebar.slider("",20, 300, 27, key="ENCODING2_DIM")
            st.sidebar.markdown('N of hidden nodes in the 2nd layer')
            latent_dim = st.sidebar.slider("",10, 20, 15, key="LATENT_DIM")
            st.sidebar.header('')
            if st.sidebar.button("Train"):
                hotel = HotelRecommendation()
                ae_embeddings = hotel.train_autoencoder(source_dict[location_option], encoding1_dim, encoding2_dim, latent_dim)
                hotel.kmeans_topic_modeling(source_dict[location_option], ae_embeddings, int(n_clusters))


    if task_option == 'Hotel Recommendation':
        if(location_option == 'AirBnb'):
            hotel = HotelRecommendation()
            hotel.hotel_recommendation()
        if((location_option != 'Select city...') & (location_option != 'AirBnb')):
            st.write("#### Input your search condition")
            # overall experience/thought
            cluster0 = st.text_input("Condition 1", 'great place to stay, recommend and encourage any tourists, good value')
            # breakfast
            cluster1 = st.text_input("Condition 2", 'breakfast included coffee, milk, cereal, bread, ham, cheese, fruit, juice')
            # room with amenities
            cluster2 = st.text_input("Condition 3", 'tv, hair dryer, kettle, microwave, cups, dishes, cutlery, cereals, shower, towels and flannels, desk, robes')
            # neighborhood and attraction
            cluster3 = st.text_input("Condition 4", 'neighbourhood filled with shops, restaurants, pubs, park')
            # host, staff
            cluster4 = st.text_input("Condition 5", 'pleasant, fast response, helpful, friendly')
            # room/flat in general conditions
            cluster5 = st.text_input("Condition 6", 'The room is clean, spacious, beautiful, furnished, full of litte details, the beds are comfy, the bathroom is clean,  garden and trees')
            # transportation
            cluster6 = st.text_input("Condition 7", 'easy access to Central London, area with easy bus, tube, underground and train connections to London Bridge, Victoria Station,Canary Wharf, london tower')
            if st.sidebar.button("Recommend"):
                hotel = HotelRecommendation()
                hotel.hotel_recommendation_booking(source_dict[location_option],
                                                   cluster0,
                                                   cluster1,
                                                   cluster2,
                                                   cluster3,
                                                   cluster4,
                                                   cluster5,
                                                   cluster6)


    if task_option == 'Knowledge Graph':
        st.sidebar.header('')
        st.sidebar.header('')
        LISTING_ID = st.sidebar.number_input(
            "Input listing_id",   
            value=2031539,  #104373
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
            entities, flag = hotel_recommendation.generate_graph_booking(source_dict[location_option], listing_id = int(LISTING_ID),cluster_id=int(cluster_option), entities=st.session_state.ENTITY_OPTION)                  
            if(flag == 1):
                st.session_state.ALL_ENTITIES = entities
            st.session_state.ENTITY_OPTION = entities
            ENTITY_OPTION = st.multiselect("Select entities", options=st.session_state.ALL_ENTITIES, default=st.session_state.ENTITY_OPTION)
        

        
        
        
            



    
