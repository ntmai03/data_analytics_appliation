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
from analysis.classification import Classification

def app():
	#st.sidebar.subheader('Select function')
	st.sidebar.subheader('')
	st.sidebar.subheader('')
	task_type = ['Select Function',
				 'Data Snapshot',
				 'Exploratory Data Analysis',
				 'Data Processing',
				 'Predictive Model',
				 'Prediction']
	task_option = st.sidebar.selectbox('', task_type)


	if task_option == 'Select Function':
		st.write('Introduction')


	#-------------------------------------------Data Snapshot-----------------------------------------
	if task_option == 'Data Snapshot':

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

	#-------------------------------------------Data Processing---------------------------------------
	if task_option == 'Data Processing':
		df = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "diabetes.csv")
		TARGET = 'Outcome'
		X_train, X_test, y_train, y_test = dm.split_data(df, df[TARGET])
		processed_X_train = data_engineering_pipeline1(X_train, train_flag=1)
		processed_X_test = data_engineering_pipeline1(X_test)
		TRAIN_VARS = list(processed_X_train.columns)

		st.write(processed_X_train.head())


	#-------------------------------------------Predictive Model---------------------------------------
	if task_option == 'Predictive Model':
		st.sidebar.subheader('')
		st.sidebar.subheader('')
		model_name = ['Logistic Regression',
					  'Decision Tree',
					  'KNN',
					  'Naive Bayes',
					  'Random Forest',
					  'Gradient Boosting Tree',
					  'SVM']
		model_option = st.sidebar.selectbox('', model_name)

		df = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "diabetes.csv")
		TARGET = 'Outcome'
		X_train, X_test, y_train, y_test = dm.split_data(df, df[TARGET])
		processed_X_train = data_engineering_pipeline1(X_train, train_flag=1)
		processed_X_test = data_engineering_pipeline1(X_test)
		TRAIN_VARS = list(processed_X_train.columns)
		model = Classification(processed_X_train, y_train,  processed_X_test, y_test, TRAIN_VARS)

		if model_option == 'Logistic Regression':
			st.write('Logistic Regression Model - Stats Model')
			result = model.train_logistict_regression_statsmodel()
			st.write(result.summary())
			st.write('--------------------------------------------------')

			st.write('Logistic Regression - Sklearn')
			result = model.train_logistict_regression_sklearn()
			st.write(result)
			st.write('--------------------------------------------------')

			st.write('Feature Selection')
			model.logistic_regression_important_feature()
			st.write('--------------------------------------------------')

			st.write('Performance Evaluation')
			model.evaluate_performance()
			st.write('--------------------------------------------------')

		if model_option == 'Decision Tree':
			st.write('Decision Tree')
			result = model.train_decision_tree()
			st.write('--------------------------------------------------')

			st.write('Performance Evaluation')
			model.evaluate_performance()
			st.write('--------------------------------------------------')	
			
		if model_option == 'Random Forest':
			st.write('Random Forest')
			result = model.train_random_forest()
			st.write('--------------------------------------------------')

			st.write('Performance Evaluation')
			model.evaluate_performance()
			st.write('--------------------------------------------------')			




	#-------------------------------------------Prediction-----------------------------------------
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