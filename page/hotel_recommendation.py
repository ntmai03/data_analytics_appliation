import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
import os

from src import config as cf
from src.util import data_manager as dm
from data_processing.booking_scrapper import BookingScrapper
from analysis.hotel_recommendation import HotelRecommendation


def app():


    ############################################### LIST OF FUNCTIONALITIES ##########################################
    task_type = [
                 "Introduction",
                 "Add new city",
                 "Collect Data", 
                 "Transform Data",
                 "Data Processing",
                 "Hotel Recommendation",
                 "Topic Modeling",
                 "Knowledge Graph"]
    task_option = st.sidebar.selectbox('Select function',task_type)
    st.sidebar.header('')


    #============================================= PART 1: Introduction ========================================
    if task_option == 'Introduction':
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Problem and Objective</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:green; font-size: 18px;"> Problems</p>', unsafe_allow_html=True)
        st.write("You are planning to go to a new city, new country to stay for a few weeks or few months and you have no idea what is where. You want to stay in a hotel, accommodation that is most suitable you such as")
        st.write("+ you like bars and clubs and near a park and an Indian restaurant because you are Indian")
        st.write("+ or I like to place in a hotel that is in quiet area, near a park or a lake, but still not far and convenient to go to the center city with many transportations, super markets restaurant near by")
        st.write(" ")
        st.write("But how can you find it since the current platform such as Booking doesn’t support such search condition. Do you wish that searching hotel booking more personalized? Like you create a profile of what you expect and the search engine will find the best hotel that matches your profile.")
        st.write(" ")
        st.write("Moreover, even you may find some proper hotels, but you are still in consideration in making decision on the final one that you’re going to stay. Reading through a lot of reviews to find out more information is tedious and time consuming. So you may want to have a general overview, general picture of what is the main thing people often review about each hotel.")
        st.markdown('<p style="color:green; font-size: 18px;"> Objectives</p>', unsafe_allow_html=True)
        st.write("This End2End application applies data science and AI techniques to address the two above problems explained in the following")
        st.write("Input their expectation about accommodations they’re looking for, then it applies NLP technique to create more personalized recommendations with specific pattern for everyone")
        st.write("The application also employs unsupervised techniques and graph network visualization to separate reviews into different subjects such as reviews related to location & neighborhood, rooms, staff & services,.. This helps users reduce time and focuses on the main things to have better understanding about the place")

        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. Main functions and Data Science Pipeline</p>', unsafe_allow_html=True)
        st.markdown('<p style="color:green; font-size: 18px;"> 1. Data pipeline (covered in functions Add new city, Collect data, Transform data)</p>', unsafe_allow_html=True)
        st.write("The application applies a simple ETL process as the following")
        st.write("**Extraction step**: It sends requests to retrieve booking’s data provided by Rapid API service. The data pipeline is built automatically, it allows user to be able to input a new city, then the application extract general info of hotels in this city and get reviews, all data is stored  in S3, each city is organized in a separate folder")
        st.write("**Transform step**: As each api response returns a json file, for each hotel there are a lot of json files for reviews data. This step merge all reviews file of each city in an csv file")
        st.write("**Load step**: The final csv file for it city is then stored in S3, this file is used as available raw  data for the analysis step")

        st.markdown('<p style="color:green; font-size: 18px;"> 2. Data Preprocessing</p>', unsafe_allow_html=True)
        st.write("**1. Split review to sentences and clauses**: First, as one review has many topics and one sentence may also include many topics, this part will also split one review to sentences and each sentences to clauses. A clause is considered as an object/ a unique for training topic model")
        st.write("**2. Text Normalization**: Second, before feature engineering, we need to pre-process, clean, and normalize text. Some of popular preprocessing techniques are applied: text tokenization, lower casting, removing special characters, removing stop words, stemming, lemmatization")
        st.write("**3. Feature Extraction using Word Embedding**: Third, for topic modeling task, Word Embedding model (Word2Vec) is employed to create semantic numeric vectors")

        st.markdown('<p style="color:green; font-size: 18px;"> 3. Hotel Recommendation</p>', unsafe_allow_html=True)
        st.write("This function first allows users to input their expectation of a house they are looking for. It then uses Bag of Words model to generate count vectors for each word in a document and hamming metric to calculate similarity between the customer’s profile and each hotel to propose the best match.")
       
        st.markdown('<p style="color:green; font-size: 18px;"> 4. Topic Modeling</p>', unsafe_allow_html=True)
        st.write("As  embedding features created by Word2Vec has 300 dimensions, dimensionality reduction technique such as AutoEncoder tries to reduce the number of dimensions to 18 in order to produce better patterns and KMeans is applies to segments reviews into different clusters which can be considered as topics. This helps users to understand different type of aspects of the hotel to help them  make better decision")

        st.markdown('<p style="color:green; font-size: 18px;"> 5. Knowledge Graph</p>', unsafe_allow_html=True)
        st.write("This function summarizes and generate a graph to depict the main patterns (key words) and their connections. This helps user to get a quick and intuitive view about main things that customers reviewed.")
        
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 4. Demo</p>', unsafe_allow_html=True)
        video_file = open('hotel_reccomendation.mp4', 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)



    #===================================== PART II:  ADD AND DISPLAY CITY LIST ===================================
    if (task_option == 'Add new city'):
        st.write("Input a city")
        new_city = st.sidebar.text_input('Input your destination').capitalize()
        # add new city to the listbox of destination
        if st.sidebar.button('Add'):
            Booking = BookingScrapper()
            cf.data['dest_name'].update({new_city:str.lower(new_city)})
            # call to booking rapidapi to get location_id and write it to config file
            try:
                dest_id, _ = Booking.search_location(str.lower(new_city))
                cf.data['dest_id'].update({new_city:dest_id})
                cf.update_yaml_config_file(cf.data)
            except:
                pass
                cf.data['dest_id'].update({new_city:''})
                cf.update_yaml_config_file(cf.data)

    # Display list of cities configured in config.yml file
    location_list = ['Select city...'] + list(cf.data['dest_name'].keys())
    location_option = st.sidebar.selectbox('',location_list)
    source_dict = cf.data['dest_name']
    location_dict = cf.data['dest_id']
    st.sidebar.header('')
    st.sidebar.header('')



    ############################################### COLLECT DATA FROM RAPID API ##########################################
    if ((task_option == 'Collect Data') & (location_option != 'Select city...')):
            Booking = BookingScrapper()

            # return the number of available hotels in Rome for a particular time
            try:
                dest_id, nr_hotels = Booking.search_location(source_dict[location_option])
                nr_hotels = int(nr_hotels)
                hotels = st.sidebar.slider("Select number of hotels",20, nr_hotels, value=60, step=20, key="N_HOTELS")
                # download hotels
                if st.sidebar.button("Download"):
                    Booking.search_accommodation(dest_id=int(dest_id), city=source_dict[location_option], nr_hotels=int(hotels))
                    Booking.get_review()
            except:
                st.write("The demo account has exceeded the MONTHLY quota for Requests to Rapid API")



    ############################################### TRANSFORM DATA ##########################################
    if ((task_option == 'Transform Data') & (location_option != 'Select city...')):
        Booking = BookingScrapper()
        Booking.merge_review_data(city=source_dict[location_option])



    ############################################### TEXT PROCESSING ##########################################
    if task_option == 'Data Processing':
        if(location_option != 'Select city...'):
            hotel = HotelRecommendation()
            hotel.text_processing(source_dict[location_option])


    ############################################### HOTEL RECOMMENDATION ##########################################
    if task_option == 'Hotel Recommendation':
        if((location_option != 'Select city...')):
            st.markdown('<p style="color:lightgreen; font-size: 30px;"> Input your search condition</p>', unsafe_allow_html=True)
            # cleaness
            condition1 = st.text_input("Condition 1", 'center, quiet, convenience')
            w1 = st.number_input("Weight 1", value=1, key='w1' )
            # food and eating
            condition2 = st.text_input("Condition 2", 'park')
            w2 = st.number_input("Weight 2", value=1, key='w2' )
            # staff and services
            condition3 = st.text_input("Condition 3", 'indian food restaurant')
            w3 = st.number_input("Weight 3", value=15, key='w3' )
            # location
            condition4 = st.text_input("Condition 4", 'friendly, helpful, tv , fan , hairdryer , fridge , microwave, kettle')
            w4 = st.number_input("Weight 4", value=1, key='w4' )
            # neighborhood
            condition5 = st.text_input("Condition 5", 'bus, train, super market, pharmacy, tesco, university')
            w5 = st.number_input("Weight 5", value=1, key='w5' )

            if st.sidebar.button("Recommend"):
                hotel = HotelRecommendation()
                result_df = hotel.hotel_recommendation_booking(source_dict[location_option],
                                                   condition1,
                                                   condition2,
                                                   condition3,
                                                   condition4,
                                                   condition5,
                                                   int(w1),int(w2),int(w3),int(w4),int(w5))
                st.write(result_df.head(3))



    ############################################### TOPIC MODELING ##########################################
    if task_option == 'Topic Modeling':
        st.markdown('<p style="color:lightgreen; font-size: 30px;"> Overview</p>', unsafe_allow_html=True)
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
            if((location_option != 'London')):
                if st.sidebar.button("Train"):
                    hotel = HotelRecommendation()
                    st.markdown('<p style="color:lightgreen; font-size: 30px;"> Training Auto Encoder to create embedding layer</p>', unsafe_allow_html=True)
                    ae_embeddings = hotel.train_autoencoder(source_dict[location_option], encoding1_dim, encoding2_dim, latent_dim, epochs)
                    st.markdown('<p style="color:lightgreen; font-size: 30px;"> KMeans clustering</p>', unsafe_allow_html=True)
                    hotel.kmeans_topic_modeling(source_dict[location_option], ae_embeddings, int(n_clusters))
            if((location_option == 'London')):
                if st.sidebar.button("Pre-trained"):
                    hotel = HotelRecommendation()
                    hotel.pretrained_model_result(source_dict[location_option])


    ############################################### KNOWLEDGE GRAPH ##########################################
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
 
        
        
        
            



    
