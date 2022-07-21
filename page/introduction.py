import streamlit as st


def app():
	st.write('This application is a demo of applying Data Science and AI techniques to build machine learning web apps for different types of Machine Learning problems as following: ')
	st.markdown('<p style="color:lightgreen; font-size: 18px;"> 1. Regression Application</p> This aplication is to analyze house data and Build predicting tool to estimate true value of a house for supporting on making investment decision ', unsafe_allow_html=True)
	st.write('')
	st.markdown('<p style="color:lightgreen; font-size: 18px;"> 2. Classification Application</p> Credit Risk Analysis is one of the most common and useful case of applying data science in order to estitmate credit risk of individuals', unsafe_allow_html=True)
	st.write('')
	st.markdown('<p style="color:lightgreen; font-size: 18px;"> 3. Finance Application</p> This application covers the whole process including collect trading data from Binance API, explore data, back testing and demo trading automatically', unsafe_allow_html=True)
	st.write('')
	st.markdown('<p style="color:lightgreen; font-size: 18px;"> 4. Unsupervised Application</p> Topic modeling is one of the application of Unsupervised techniques. It is used to extract and depict key themes or concepts which are latent in a large corpora of text documents which are represented as topics', unsafe_allow_html=True)
	st.write('')
	st.markdown('<p style="color:lightgreen; font-size: 18px;"> 5. NLP Application </p> This is a  NLP application of Hotel booking listings to provide personalized search and discover info from hotel reviews. It includes 2 parts: Part 1 collects hotel review data from Booking API, transform and store final data on S3. Part 2 applies data mining techniques such as topic modeling, recommendation and information retrievals to build a machine learning app', unsafe_allow_html=True)
	st.write('')
	st.markdown('<p style="color:lightgreen; font-size: 18px;"> 6. Online shopping Recommender System </p> Surprising people by recommending them products, movies, hotels as per their choice is what makes recommender systems so useful and close to our day-to-day lives. Recommender System is a way of modeling and rearranging information available about user references and then using this information to provide informed recommendations on the basic of that information', unsafe_allow_html=True)
