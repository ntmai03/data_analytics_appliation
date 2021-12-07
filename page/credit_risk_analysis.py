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
from pipeline.credit_risk_pipeline import *
from analysis.credit_risk import CreditRisk

def app():
	#st.sidebar.subheader('Select function')
	st.sidebar.subheader('')
	st.sidebar.subheader('')
	task_type = ['Select Function',
				 'Data Snapshot',
				 'Exploratory Data Analysis',
				 'Data Processing',
				 'Train Model',
				 'Make Prediction']
	task_option = st.sidebar.selectbox('', task_type)


	if task_option == 'Select Function':
		st.write('Introduction')


	#============================================= Data Snapshot========================================
	if task_option == 'Data Snapshot':
		# get data from s3
		credit_risk = CreditRisk()
		credit_risk.load_dataset()
		st.write("#### First 10 rows")
		st.write(credit_risk.data[credit_risk.SELECTED_VARS].head(10))
	#============================================= Data Snapshot========================================



	#=========================================== Data Processing =======================================
	if task_option == 'Data Processing':
		credit_risk = CreditRisk()
		st.write("#### Raw data")
		#credit_risk.load_dataset()
		#st.write(credit_risk.X_train[credit_risk.SELECTED_VARS].head(10))

		st.write("#### Prepocessed data")
		credit_risk.prepare_dataset()
		st.write(credit_risk.processed_X_train.head(10))
	#=========================================== Data Processing =======================================



	#========================================== Predictive Model =======================================
	if task_option == 'Train Model':
		st.sidebar.subheader('')
		st.sidebar.subheader('')
		model_name = ['Select model',
					  'Logistic Regression',
					  'Decision Tree',
					  'KNN',
					  'Naive Bayes',
					  'Random Forest',
					  'Gradient Boosting Tree',
					  'Xgboost',
					  'DNN']
		model_option = st.sidebar.selectbox('', model_name)

		if model_option == 'Logistic Regression':
			credit_risk = CreditRisk()
			st.write('Logistic Regression')
			credit_risk.train_logistic_regression_statsmodel()
			st.write('--------------------------------------------------')
			credit_risk.train_logistic_regression_sklearn()
		
		if model_option == 'Decision Tree':
			credit_risk = CreditRisk()
			st.write('Decision Tree')
			credit_risk.train_decision_tree()

		if model_option == 'Random Forest':
			credit_risk = CreditRisk()
			st.write('Random Forest')
			credit_risk.train_random_forest()

		if model_option == 'Xgboost':
			credit_risk = CreditRisk()
			st.write('Xgboost')
			credit_risk.train_xgboost()

		if model_option == 'KNN':
			credit_risk = CreditRisk()
			st.write('KNN')
			credit_risk.train_knn()

		if model_option == 'Gradient Boosting Tree':
			credit_risk = CreditRisk()
			st.write('Gradient Boosting Tree')
			credit_risk.train_gradient_boosting()

		if model_option == 'Naive Bayes':
			credit_risk = CreditRisk()
			st.write('GaussianNB')
			credit_risk.train_GaussianNB()
	#========================================== Predictive Model =======================================



	#============================================= Prediction ==========================================
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
	#============================================= Prediction ==========================================