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

    ############################################### ADD A NEW CITY TO LISTBOX ##########################################
    st.sidebar.header('')
    st.sidebar.header('')  
    new_city = st.sidebar.text_input('Input your destination').capitalize()
    # add new city to the listbox of destination
    if st.sidebar.button('Add'):
        Booking = BookingScrapper()
        cf.data['dest_name'].update({new_city:str.lower(new_city)})
        # call to booking rapidapi to get location_id and write it to config file
        dest_id, _ = Booking.search_location(str.lower(new_city))
        cf.data['dest_id'].update({new_city:dest_id})
        cf.update_yaml_config_file(cf.data)

    # Display list of cities configured in config.yml file
    location_list = ['Select city...'] + list(cf.data['dest_name'].keys())
    location_option = st.sidebar.selectbox('',location_list)
    source_dict = cf.data['dest_name']
    location_dict = cf.data['dest_id']
    st.sidebar.header('')
    st.sidebar.header('')



    ############################################### LIST OF FUNCTIONALITIES ##########################################
    task_type = ["Select function",
                 "Collect Data", 
                 "Transform Data",
                 "Data Processing",
                 "Topic Modeling",
                 "Hotel Recommendation",
                 "Knowledge Graph"]
    task_option = st.sidebar.selectbox('',task_type)
    st.sidebar.header('')


    ############################################### COLLECT DATA FROM RAPID API ##########################################
    if ((task_option == 'Collect Data') & (location_option != 'Select city...')):
            Booking = BookingScrapper()

            # return the number of available hotels in Rome for a particular time
            _, nr_hotels = Booking.search_location(source_dict[location_option])
            nr_hotels = int(nr_hotels)
            hotels = st.sidebar.slider("Select number of hotels",20, nr_hotels, value=60, step=20, key="N_HOTELS")

            # download hotels
            if st.sidebar.button("Download"):
                Booking.search_accommodation(dest_id=int(location_dict[location_option]), city=source_dict[location_option], nr_hotels=int(hotels))
                Booking.get_review()


    ############################################### TRANSFORM DATA ##########################################
    if ((task_option == 'Transform Data') & (location_option != 'Select city...')):
            Booking = BookingScrapper()
            Booking.create_review_file(city=source_dict[location_option])


    ############################################### TEXT PROCESSING ##########################################
    if task_option == 'Data Processing':
        if(location_option != 'Select city...'):
            st.write("There are two steps for feature engineering on text data: (1) Preprocessing and normalizing text, (2) Feature extraction and engineering")
            st.write("**Step 1 - Preprocessing and normalizing text**:") 
            st.write("+ First, as one review has many topics and one sentence may also include many topics, this part will also split one review to sentences and each sentences to clauses. A clause is considered as an object/ a unique for training topic model.")
            st.write("+ Second, some of popular preprocessing techniques are applied: text tokenization, lower casting, removing special characters, removing stop words, stemming, lemmatization")
            st.write("**Step 2 - Text to Feature**: For transforming text to numbers, Word Embedding model is employed to create semantic numeric vectors")
            hotel = HotelRecommendation()
            hotel.text_processing(source_dict[location_option])


    if task_option == 'Topic Modeling':
        st.write("There are two steps for Topic Modeling: (1) Dimensionality Reduction, (2) Clustering")
        st.write("**Step 1 - Dimensionality Reduction**: In this step, Auto Enconder Decoder is employed to create a representation layer which has fewer features and relationships for easy exploration, visualization, and also less likely to overfit training model")
        st.write("**Step 2  - K-Means Clustering**: After reducing features and visualization, KMeans is chosen for topic modeling technique")
        if((location_option != 'Select city...')):
            st.sidebar.markdown('N of clusters')
            n_clusters = st.sidebar.slider("",2, 10, 8, key="N_CLUSTERS")
            st.sidebar.markdown('N of hidden nodes in the 1st layer')
            encoding1_dim = st.sidebar.slider("",50, 300, 50, key="ENCODING1_DIM")
            st.sidebar.markdown('N of hidden nodes in the 2nd layer')
            encoding2_dim = st.sidebar.slider("",20, 800, 600, key="ENCODING2_DIM")
            st.sidebar.markdown('N of hidden nodes in the 2nd layer')
            latent_dim = st.sidebar.slider("",10, 20, 15, key="LATENT_DIM")
            st.sidebar.markdown('Num of epochs')
            epochs = st.sidebar.slider("",10, 80, 30, key="EPOCHS")
            st.sidebar.header('')
            if st.sidebar.button("Train"):
                hotel = HotelRecommendation()
                st.markdown('<p style="color:Green; font-size: 30px;"> 1. Training Auto Encoder to create embedding layer</p>', unsafe_allow_html=True)
                ae_embeddings = hotel.train_autoencoder(source_dict[location_option], encoding1_dim, encoding2_dim, latent_dim, epochs)
                st.markdown('<p style="color:Green; font-size: 30px;"> 2. KMeans clustering</p>', unsafe_allow_html=True)
                hotel.kmeans_topic_modeling(source_dict[location_option], ae_embeddings, int(n_clusters))


    if task_option == 'Hotel Recommendation':
        if((location_option != 'Select city...')):
            st.markdown('<p style="color:Green; font-size: 30px;"> Input your search condition</p>', unsafe_allow_html=True)
            # cleaness
            condition1 = st.text_input("Condition 1", 'clean , spacious and really easy to find')
            # food and eating
            condition2 = st.text_input("Condition 2", "eggs , bacon , toast , cereal , coffee & tea in our apartment fridge")
            # staff and services
            condition3 = st.text_input("Condition 3", "staff very friendly and helpful")
            # location
            condition4 = st.text_input("Condition 4", "close to underground cannon st station , bank station")
            # neighborhood
            condition5 = st.text_input("Condition 5", "location in central london is close to at least 3 tube stations , restaurants , grocery stores")
            # bathroom
            condition6 = st.text_input("Condition 6", "large bathroom with large sheet towels and white company toiletries, body wash & lotion , shampoo & conditioner")

            if st.sidebar.button("Recommend"):
                hotel = HotelRecommendation()
                hotel.hotel_recommendation_booking(source_dict[location_option],
                                                   condition1,
                                                   condition2,
                                                   condition3,
                                                   condition4,
                                                   condition5,
                                                   condition6)


    if task_option == 'Knowledge Graph':
        st.sidebar.header('')
        st.sidebar.header('')
        LISTING_ID = st.sidebar.number_input(
            "Input listing_id",   
            value=1876697,  #Rome
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

        NODE_DEGREE = st.sidebar.number_input(
            "Input Node Degree",   
            value=0,  #104373
            key="NODE_DEGREE",
        )


        if st.sidebar.button('Generate Graph'):
            hotel_recommendation = HotelRecommendation()
            hotel_recommendation.generate_graph_booking(source_dict[location_option], listing_id = int(LISTING_ID),cluster_id=int(cluster_option), node_degree = int(NODE_DEGREE))                  
 
        
        
        
            



    
