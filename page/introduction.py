import streamlit as st


def app():


	st.markdown('<p style="color:lightgreen; font-size: 25px;"> Overview </p>', unsafe_allow_html=True)
	st.write('This application is a demo of applying Data Science and AI techniques to build machine learning web apps for different types of Machine Learning problems as following: ')
	st.markdown('<p style="color:green; font-size: 18px;"> 1. Regression Application</p> This aplication is to analyze house data and Build predicting tool to estimate true value of a house for supporting on making investment decision ', unsafe_allow_html=True)
	st.write('')
	st.markdown('<p style="color:green; font-size: 18px;"> 2. Classification Application</p> Credit Risk Analysis is one of the most common and useful case of applying data science in order to estitmate credit risk of individuals', unsafe_allow_html=True)
	st.write('')
	st.markdown('<p style="color:green; font-size: 18px;"> 3. Finance Application</p> This application covers the whole process including collect trading data from Binance API, explore data, back testing and demo trading automatically', unsafe_allow_html=True)
	st.write('')
	st.markdown('<p style="color:green; font-size: 18px;"> 4. Unsupervised Application</p> Topic modeling is one of the application of Unsupervised techniques. It is used to extract and depict key themes or concepts which are latent in a large corpora of text documents which are represented as topics', unsafe_allow_html=True)
	st.write('')
	st.markdown('<p style="color:green; font-size: 18px;"> 5. NLP Application </p> This is a  NLP application of Hotel booking listings to provide personalized search and discover info from hotel reviews. It includes 2 parts: Part 1 collects hotel review data from Booking API, transform and store final data on S3. Part 2 applies data mining techniques such as topic modeling, recommendation and information retrievals to build a machine learning app', unsafe_allow_html=True)
	st.write('')
	st.markdown('<p style="color:green; font-size: 18px;"> 6. Online shopping Recommender System </p> Surprising people by recommending them products, movies, hotels as per their choice is what makes recommender systems so useful and close to our day-to-day lives. Recommender System is a way of modeling and rearranging information available about user references and then using this information to provide informed recommendations on the basic of that information', unsafe_allow_html=True)

	st.write('')
	st.write('')

	st.markdown('<p style="color:lightgreen; font-size: 25px;"> Data and Machine Learning Pipeline </p>', unsafe_allow_html=True)
	st.write("The analysis applied  standard process for data mining (CRISP-DM) model which includes phases: ")
	st.markdown('<p style="color:green; font-size: 18px;"> 1. Identify Problem</p> This step is to define business problem and business objective with desired outcome into Analytic terms', unsafe_allow_html=True)
	st.write(' ')
	st.markdown('<p style="color:green; font-size: 18px;"> 2. Data Collection and Preparation</p> This phase involves collecting the data, describing the various attributes, preparing data into right format for analysis and do tasks such as cleaning data, define variables, joining/appending multiple datasets. The purpose of this step is to construct the final dataset from original raw dataset as optimum input into the modeling algorithms. This phase is important because bad data or insufficient knowledege about available data can have cascading adverse effects in the later stages of the analysis', unsafe_allow_html=True)
	st.write(' ')
	st.markdown('<p style="color:green; font-size: 18px;"> 3. Exploratory data analysis (EDA) </p> EDA is a crucial step in the lifecycle. Here, the main objective is to explore and understand the characteristics, patterns, nature of data as well as the trustworthiness of the data. The common tools for this are descriptive statistics, charts and visualizations to look at various data attributes, find associations and correlations and make a note of data quality problems if any', unsafe_allow_html=True)
	st.write(' ')
	st.markdown('<p style="color:green; font-size: 18px;"> 4. Data Preprocessing</p> Data preprocessing is one of the most important steps and critical in success of machine learning techniques. Data Preprocessing is performed by techniques such as handling missing values, outlier data, applying transformation methods, binning approach, creating derived features…', unsafe_allow_html=True)
	st.write(' ')
	st.markdown('<p style="color:green; font-size: 18px;"> 5. Modeling</p> To develop more accurate prediction models with high-performance capacity, advanced computational techniques and data mining methods could be employed. The focus of machine learning and data mining (DM) techniques are developing computerized and efficient predictive modeling by exploring hidden and unknown patterns in data to discover knowledge with high accuracy', unsafe_allow_html=True)
	st.write(' ')
	st.markdown('<p style="color:green; font-size: 18px;"> 6. Model Evaluation</p> At this stage, after prediction model development base on prospective data, the performance of model should be evaluated for unseen data to validate for real-world settings. ', unsafe_allow_html=True)
	st.write(' ')
	st.markdown('<p style="color:green; font-size: 18px;"> 7. Model Deployment</p> The deployment of machine learning models is the process for making models available in production environments, where they can provide predictions to other software systems', unsafe_allow_html=True)
	st.write(' ')
