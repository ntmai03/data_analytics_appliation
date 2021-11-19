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
from pipeline.diabetes_pipeline import *

def app():
	st.sidebar.subheader('Select function')
	task_type = ['Introduction',
				 'Exploratory Data Analysis',
				 'Data Processing',
				 'Predictive Model',
				 'Prediction']
	task_option = st.sidebar.selectbox('', task_type)
	st.sidebar.header('')

	if task_option == 'Introduction':
		# get data from s3
		df = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "diabetes.csv")
		st.write("#### First 100 rows")
		st.write(df.head(100))

		# get data from local machine
		'''
		data_file = os.path.join(cf.DATA_RAW_PATH, "diabetes.csv")
		df = dm.load_csv_data(data_file)
		st.write("#### First 100 rows")
		st.write(df.head(100))
		'''


	if task_option == 'Prediction':
		st.write("#### Input your data for prediction")
		Pregnancies = st.text_input("Pregnancies", '3')
		Glucose = st.text_input("Glucose", '158')
		BloodPressure = st.text_input("BloodPressure", '76')
		SkinThickness = st.text_input("SkinThickness", '36')
		Insulin = st.text_input("Insulin", '245')
		BMI = st.text_input("BMI", '31.6')
		DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction", '0.851')
		Age = st.text_input("Age", '28')

		# Add button Predict
		if st.button("Predict"):
			new_obj = dict({'Outcome':1,
							'Pregnancies':[Pregnancies],
							'Glucose':[Glucose],
							'BloodPressure':[BloodPressure],
							'SkinThickness':[SkinThickness],
							'Insulin':[Insulin],
							'BMI':[BMI],
							'DiabetesPedigreeFunction':[DiabetesPedigreeFunction],
							'Age':[Age] })
			new_obj = pd.DataFrame.from_dict(new_obj)
			st.write(new_obj)
			new_obj = data_engineering_pipeline1(new_obj)
			model_file = os.path.join(cf.TRAINED_MODEL_PATH,"diabetes_logistic_regression.pkl")
			model = joblib.load(model_file)
			st.write("**Predicted Diabetes**: ", model.predict(new_obj))