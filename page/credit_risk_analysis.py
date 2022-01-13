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
		credit_risk.load_dataset()
		st.write(credit_risk.data[credit_risk.SELECTED_VARS].head(10))
		st.write(credit_risk.data_processing_pipeline(credit_risk.X_train, 1).head(10))

		processed_X_train = credit_risk.data_processing_pipeline(credit_risk.X_train, 1)
		df_train = pd.concat([processed_X_train, credit_risk.y_train], axis=1)
		df_train.to_csv('credit_risk_train.csv', index=False)
		processed_X_test = credit_risk.data_processing_pipeline(credit_risk.X_train, 0)
		df_test = pd.concat([processed_X_test, credit_risk.y_test], axis=1)
		df_test.to_csv('credit_risk_test.csv', index=False)

	#=========================================== Data Processing =======================================



	#========================================== Predictive Model =======================================
	if task_option == 'Train Model':
		st.sidebar.subheader('')
		st.sidebar.subheader('')
		model_name = ['Select model',
					  'Logistic Regression',
					  'Lasso Regression',
					  'Decision Tree',
					  'KNN',
					  'Naive Bayes',
					  'LDA',
					  'QDA',					
					  'Random Forest',
					  'Gradient Boosting Tree',
					  'Xgboost',
					  'DNN']
		model_option = st.sidebar.selectbox('', model_name)

		if model_option == 'Logistic Regression':
			threshold = st.sidebar.slider("",0.05, 0.95, 0.5, key="THRESHOLD")
			if st.sidebar.button("Select threshold"):
				credit_risk = CreditRisk()
				#credit_risk.train_logistic_regression_statsmodel()
				#st.write('--------------------------------------------------')
				credit_risk.logistic_regression_analysis(threshold)

		if model_option == 'Lasso Regression':
			credit_risk = CreditRisk()
			credit_risk.lasso_analysis()
		
		if model_option == 'Decision Tree':
			st.sidebar.markdown('max_depth')
			max_depth = st.sidebar.slider("",1, 50, 8, key="MAX_DEPTH")
			st.sidebar.markdown('max_features')
			max_features = st.sidebar.slider("",5, 20, 10, key="MAX_FEATURES")
			st.sidebar.markdown('min_samples_leaf')
			min_samples_leaf = st.sidebar.slider("",10, 50, 30, key="MIN_SAMPLES_LEAF")
			st.sidebar.header('')
			if st.sidebar.button("Train"):
				credit_risk = CreditRisk()
				credit_risk.decision_tree_analysis(max_depth, max_features, min_samples_leaf)

		if model_option == 'Random Forest':
			st.sidebar.markdown('max_depth')
			max_depth = st.sidebar.slider("",1, 50, 8, key="MAX_DEPTH")
			st.sidebar.markdown('max_features')
			max_features = st.sidebar.slider("",5, 20, 10, key="MAX_FEATURES")
			st.sidebar.markdown('min_samples_leaf')
			min_samples_leaf = st.sidebar.slider("",10, 50, 30, key="MIN_SAMPLES_LEAF")
			st.sidebar.header('')
			if st.sidebar.button("Train"):
				credit_risk = CreditRisk()
				credit_risk.random_forest_analysis(max_depth, max_features, min_samples_leaf)

		if model_option == 'Xgboost':
			st.sidebar.markdown('max_depth')
			max_depth = st.sidebar.slider("",1, 50, 8, key="MAX_DEPTH")
			st.sidebar.markdown('max_features')
			max_features = st.sidebar.slider("",5, 20, 10, key="MAX_FEATURES")
			st.sidebar.markdown('min_samples_leaf')
			min_samples_leaf = st.sidebar.slider("",10, 50, 30, key="MIN_SAMPLES_LEAF")
			st.sidebar.header('')
			if st.sidebar.button("Train"):
				credit_risk = CreditRisk()
				credit_risk.xgboost_analysis(max_depth, max_features, min_samples_leaf)

		if model_option == 'Gradient Boosting Tree':
			st.sidebar.markdown('max_depth')
			max_depth = st.sidebar.slider("",1, 50, 8, key="MAX_DEPTH")
			st.sidebar.markdown('max_features')
			max_features = st.sidebar.slider("",5, 20, 10, key="MAX_FEATURES")
			st.sidebar.markdown('min_samples_leaf')
			min_samples_leaf = st.sidebar.slider("",10, 50, 30, key="MIN_SAMPLES_LEAF")
			st.sidebar.header('')
			if st.sidebar.button("Train"):
				credit_risk = CreditRisk()
				credit_risk.gradient_boosting_analysis(max_depth, max_features, min_samples_leaf)

		if model_option == 'KNN':
			st.sidebar.markdown('n_neighbors')
			n_neighbors = st.sidebar.slider("",10, 20, 10, key="N_NEIGHBORS")
			st.sidebar.header('')
			if st.sidebar.button("Train"):
				credit_risk = CreditRisk()
				credit_risk.knn_analysis(n_neighbors)

		if model_option == 'Naive Bayes':
			credit_risk = CreditRisk()
			credit_risk.GaussianNB_analysis()

		if model_option == 'LDA':
			credit_risk = CreditRisk()
			credit_risk.lda_analysis()

		if model_option == 'QDA':
			credit_risk = CreditRisk()
			credit_risk.qda_analysis()
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