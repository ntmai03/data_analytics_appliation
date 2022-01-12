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
from analysis.shopping_recommendation import ShoppingRecommendation

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



	if task_option == 'Data Processing':
		shopping = ShoppingRecommendation()
		shopping.prepare_dataset()
		st.write(shopping.transaction_detail.head())



		

	#-------------------------------------------Predictive Model---------------------------------------
	if task_option == 'Predictive Model':
		st.sidebar.subheader('')
		st.sidebar.subheader('')
		model_name = ['Select model...',
					  'Item_Item_CF']
		model_option = st.sidebar.selectbox('', model_name)
		if model_option == 'Item_Item_CF':
			shopping = ShoppingRecommendation()
			shopping.ItemItemCF(710)


