import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
import os
from io import BytesIO

sys.path.append('src')
from src import config as cf
from src.util import data_manager as dm
from src.data_processing import kchouse_feature_engineering as fe
from pipeline.kchouse_pipeline import *
from analysis.house_price import HousePrice

def app():
	st.sidebar.subheader('Select function')
	task_type = ['Introduction',
				 'Data Snapshot',
				 'Exploratory Data Analysis',
				 'Data Processing',
				 'Predictive Model',
				 'Prediction']
	task_option = st.sidebar.selectbox('', task_type)
	st.sidebar.header('')

	if task_option == 'Introduction':
		# https://docs.streamlit.io/library/get-started/create-an-app
		st.write('Introduction')	



	if task_option == 'Data Snapshot':
		# get data from s3
		houseprice = HousePrice()
		houseprice.load_dataset()
		st.write("#### First 10 rows")
		#df = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "kc_house_data.csv")
		st.write(houseprice.data[houseprice.SELECTED_VARS].head(10))
		st.write(houseprice.data_processing_pipeline(houseprice.X_train, 1).head(10))

		houseprice.processed_X_train = houseprice.data_processing_pipeline(houseprice.X_train, 1)
		# calculate the correlations using pandas corr and round the values to 2 decimals
		corr_mat = houseprice.processed_X_train.corr().round(2)
		# plot the correlation matrix using seaorn annot=True to print the correlation values inside the squares

	
		st.write("")	
		col1, col2, col3 = st.columns([1, 6, 1])
		fig, ax = plt.subplots(figsize=(15, 15))
		sns.heatmap(data=corr_mat, annot=True, ax=ax)
		ax.set_title("Correlation Matrix")
		buf = BytesIO()
		fig.savefig(buf, format="png")
		col2.image(buf)


		

	#-------------------------------------------Predictive Model---------------------------------------
	if task_option == 'Predictive Model':
		st.sidebar.subheader('')
		st.sidebar.subheader('')
		model_name = ['Select model...',
					  'Linear Regression',
					  'Decision Tree',
					  'Random Forest',
					  'Gradient Boosting Tree']
		'''			  
		houseprice = HousePrice()
		houseprice.load_dataset()
		houseprice.processed_X_train = houseprice.data_processing_pipeline(houseprice.X_train, 1)
		houseprice.processed_X_test = houseprice.data_processing_pipeline(houseprice.X_test, 0)	
		'''

		model_option = st.sidebar.selectbox('', model_name)

		if model_option == 'Linear Regression':
			houseprice = HousePrice()
			st.write('Linear Regression Model - Stats Model')
			result = houseprice.train_regression_statsmodel()
			st.write(result.summary())
			st.write('--------------------------------------------------')

		if model_option == 'Decision Tree':
			st.sidebar.markdown('max_depth')
			max_depth = st.sidebar.slider("",1, 50, 8, key="MAX_DEPTH")
			st.sidebar.markdown('max_features')
			max_features = st.sidebar.slider("",5, 20, 10, key="MAX_FEATURES")
			st.sidebar.markdown('min_samples_leaf')
			min_samples_leaf = st.sidebar.slider("",10, 50, 30, key="MIN_SAMPLES_LEAF")
			st.sidebar.header('')
			if st.sidebar.button("Train"):
				houseprice = HousePrice()
				houseprice.decision_tree_analysis(max_depth, max_features, min_samples_leaf)

		if model_option == 'Random Forest':
			st.sidebar.markdown('max_depth')
			max_depth = st.sidebar.slider("",1, 20, 8, key="MAX_DEPTH")
			st.sidebar.markdown('max_features')
			max_features = st.sidebar.slider("",5, 20, 10, key="MAX_FEATURES")
			st.sidebar.markdown('min_samples_leaf')
			min_samples_leaf = st.sidebar.slider("",50, 200, 30, key="MIN_SAMPLES_LEAF")
			st.sidebar.header('')
			if st.sidebar.button("Train"):
				houseprice = HousePrice()
				houseprice.random_forest_analysis(max_depth, max_features, min_samples_leaf)


		if model_option == 'Gradient Boosting Tree':
			st.sidebar.markdown('max_depth')
			max_depth = st.sidebar.slider("",1, 20, 8, key="MAX_DEPTH")
			st.sidebar.markdown('max_features')
			max_features = st.sidebar.slider("",5, 20, 10, key="MAX_FEATURES")
			st.sidebar.markdown('min_samples_leaf')
			min_samples_leaf = st.sidebar.slider("",50, 200, 30, key="MIN_SAMPLES_LEAF")
			st.sidebar.header('')
			if st.sidebar.button("Train"):
				houseprice = HousePrice()
				houseprice.gbt_analysis(max_depth, max_features, min_samples_leaf)


	if task_option == 'Prediction':
	    st.write("#### Input your data for prediction")
	    bedrooms = st.text_input("Num of bedrooms", '3')
	    bathrooms = st.text_input("Num of bathrooms", '1')
	    sqft_living = st.text_input("sqft_living", '1180')
	    sqft_lot = st.text_input("sqft_lot", '5650')
	    floors = st.text_input("floors", '1')
	    waterfront = st.text_input("waterfront", '0')
	    view = st.text_input("view", '0')
	    condition = st.text_input("condition", '3')
	    grade = st.text_input("grade", '7')
	    sqft_above = st.text_input("sqft_above", '1180')
	    sqft_basement = st.text_input("sqft_basement", '0')
	    yr_built = st.text_input("yr_built", '1955')
	    yr_renovated = st.text_input("yr_renovated", '0')
	    lat_corr = st.text_input("lat", '47.5112')
	    long_corr = st.text_input("long", '-122.2570')
	    sqft_living15 = st.text_input("sqft_living15", '1340')
	    sqft_lot15 = st.text_input("sqft_lot15", '5650')
	    date = st.text_input("date", '20141013T000000')
	    zipcode = st.text_input("zipcode", '98178')

	    if st.button("Predict"):
	    	new_obj = dict({'price': 221900.0,
	    					'bedrooms':[bedrooms],
	    					'bathrooms':[bathrooms],
	    					'sqft_living':[sqft_living],
	    					'sqft_lot':[sqft_lot],
	    					'floors':[floors],
	    					'waterfront':[waterfront],
	    					'view':[view],
	    					'condition':[condition],
	    					'grade':[grade],
	    					'sqft_above':[sqft_above],
	    					'sqft_basement':[sqft_basement],
	    					'yr_built':[yr_built],
	    					'yr_renovated':[yr_renovated],
	    					'lat':[lat_corr],
	    					'long':[long_corr],
	    					'sqft_living15':[sqft_living15],
	    					'sqft_lot15':[sqft_lot15],
	    					'zipcode':[zipcode],
	    					'date': [date]
	    		})
	    	new_obj = pd.DataFrame.from_dict(new_obj)
	    	st.write(new_obj)
	    	new_obj = data_engineering_pipeline1(new_obj)
	    	model_file = os.path.join(cf.TRAINED_MODEL_PATH, "kchouse_xgb.pkl")
	    	model = joblib.load(model_file)
	    	st.write("**Predicted Price**: ", np.exp(model.predict(new_obj)))