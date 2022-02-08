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
		processed_X_test = credit_risk.data_processing_pipeline(credit_risk.X_test, 0)
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
	if task_option == 'Make Prediction':
		st.write("#### Input your data for prediction")
		loan_amnt = st.text_input("loan_amnt", '5000')
		term = st.text_input("term", ' 36 months')
		int_rate = st.text_input("int_rate", '12.49')
		grade = st.text_input("grade", 'B')
		sub_grade = st.text_input("sub_grade", 'B5')
		emp_title = st.text_input("emp_title", 'author')
		emp_length = st.text_input("emp_length", '1 year')
		home_ownership = st.text_input("home_ownership", 'OWN')
		annual_inc = st.text_input("annual_inc", '28000.0')
		verification_status = st.text_input("verification_status", 'Not Verified')
		issue_d = st.text_input("issue_d", 'Oct-14')
		loan_status = st.text_input("loan_status", '5000')
		purpose = st.text_input("purpose", 'medical')
		title = st.text_input("title", 'Medical expenses')
		zip_code = st.text_input("zip_code", '480xx')
		addr_state = st.text_input("addr_state", 'MI')
		dti = st.text_input("dti", '20.87')
		delinq_2yrs = st.text_input("delinq_2yrs", '0.0')
		earliest_cr_line = st.text_input("earliest_cr_line", 'Dec-99')
		inq_last_6mths = st.text_input("inq_last_6mths", '1.0')
		open_acc = st.text_input("open_acc", '8.0')
		pub_rec = st.text_input("pub_rec", '0.0')
		revol_bal = st.text_input("revol_bal", '4549')
		revol_util = st.text_input("revol_util", '64.1')
		total_acc = st.text_input("total_acc", '18.0')		
		initial_list_status = st.text_input("initial_list_status", 'w')


		# Add button Predict
		if st.button("Predict"):
			new_obj = dict({
							'loan_amnt':[loan_amnt],
							'term':[term],
							'int_rate':[int_rate],
							'grade':[grade],
							'sub_grade':[sub_grade],
							'emp_title':[emp_title],
							'emp_length':[emp_length],
							'home_ownership':[home_ownership],
							'annual_inc':[annual_inc],
							'verification_status':[verification_status],
							'issue_d':[issue_d],
							'loan_status':[loan_status],
							'purpose':[purpose],
							'title':[title],
							'zip_code':[zip_code],
							'addr_state':[addr_state],
							'dti':[dti],
							'delinq_2yrs':[delinq_2yrs],
							'earliest_cr_line':[earliest_cr_line],
							'inq_last_6mths':[inq_last_6mths],
							'open_acc':[open_acc],
							'pub_rec':[pub_rec],
							'revol_bal':[revol_bal],
							'revol_util':[revol_util],
							'total_acc':[total_acc],
							'initial_list_status':[initial_list_status]
							})
			new_obj = pd.DataFrame.from_dict(new_obj)
			st.write(new_obj)
			credit_risk = CreditRisk()
			new_obj = credit_risk.data_processing_pipeline(new_obj, 0)
			st.write(new_obj)
			model_file = os.path.join(cf.TRAINED_MODEL_PATH,"credit_risk_logistic_regression.pkl")
			model = joblib.load(model_file)
			st.write("**Predicted Default**: ", model.predict(new_obj))
	#============================================= Prediction ==========================================