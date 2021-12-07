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
from analysis.titanic_ds import Titanic

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
		titanic = Titanic()
		titanic.prepare_dataset()
		st.write("#### First 100 rows")
		st.write(titanic.data.head(30))


	#-------------------------------------------Data Processing---------------------------------------
	if task_option == 'Data Processing':
		titanic = Titanic()
		titanic.prepare_dataset()
		st.write('Raw data')
		st.write(titanic.X_train[titanic.SELECTED_VARS].head(10))
		st.write('Processed data')
		st.write(titanic.data_processing_pipeline(titanic.X_train, 1).head(10))
		st.write(titanic.data_processing_pipeline(titanic.X_train, 1).isnull().sum())


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
		titanic = Titanic()
		titanic.prepare_dataset()
		titanic.processed_X_train = titanic.data_processing_pipeline(titanic.X_train, 1)
		titanic.processed_X_test = titanic.data_processing_pipeline(titanic.X_test, 1)	
		
		model_option = st.sidebar.selectbox('', model_name)

		
		if model_option == 'Logistic Regression':
			st.write('Logistic Regression Model - Stats Model')
			result = titanic.train_logistict_regression_statsmodel()
			st.write(result.summary())
			st.write('--------------------------------------------------')

			st.write('Logistic Regression - Sklearn')
			#result = titanic.train_logistict_regression_sklearn()
			st.write(result)
			st.write('--------------------------------------------------')

			st.write('Feature Selection')
			#titanic.logistic_regression_important_feature()
			st.write('--------------------------------------------------')

			st.write('Performance Evaluation')
			#titanic.evaluate_performance()
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