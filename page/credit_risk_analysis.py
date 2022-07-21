import streamlit as st
import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
import os

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

sys.path.append('src')
from src import config as cf
from src.util import data_manager as dm
from analysis.credit_risk import CreditRisk
from src.util import classification_util as clfu

def app():
	st.sidebar.subheader('')
	st.sidebar.subheader('Select Function')
	task_type = ['Introduction',
				 'Data Understanding',
				 'Exploratory Data Analysis',
				 'Data Processing',
				 'Train Model',
				 'Make Prediction']
	task_option = st.sidebar.selectbox('', task_type)


	#============================================= Introduction ========================================
	if task_option == 'Introduction':
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> Business Understanding</p>', unsafe_allow_html=True)
		st.write("LendingClub is a US peer-to-peer lending company, headquartered in San Francisco, California. It was the first peer-to-peer lender to register its offerings as securities with the Securities and Exchange Commission (SEC), and to offer loan trading on a secondary market. LendingClub is the world's largest peer-to-peer lending platform")
		st.write("Given historical data on loans with information on whether or not the borrower defaulted (charge-off), can we build a model that can predict whether or not a borrower will pay back their loan? This way in the future when we get a new potential customer we can access whether or not they are likely to pay back the loan.")
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> Defining Problem</p>', unsafe_allow_html=True)
		st.write("It is a classificaion problem, the project is to employ Machine Learning algorithms to build the predictive model that allows to predict default rate or credit risk probability of a given credit profile")


	#============================================= Data Understanding ========================================
	if task_option == 'Data Understanding':
		# get data from s3
		credit_risk = CreditRisk()

		st.markdown('<p style="color:Green; font-size: 25px;"> 1. Data Collection</p>', unsafe_allow_html=True)
		st.write("There are many LendingClub data sets on Kaggle. This project uses a subset of the Lending Club datatset obtained from Kaggle: https://www.kaggle.com/wordsforthewise/lending-club")
		
		# Raw data
		credit_risk.load_dataset()
		st.markdown('<p style="color:Green; font-size: 25px;"> 2. Data snapshot of the first 10 rows</p>', unsafe_allow_html=True)
		st.write(credit_risk.data.head(10))
		st.write('The dataset has 466285 rows, 75 columns')
		st.write("There are fields that are not necessary for building predictive models such as Unnamed, id, member_id, url, desc. These fields should be excluded from the final dataset")
		st.write("There are many fields describing the payment process during a loan period such as open_il_6m, open_il_12m, open_il_24m, mths_since_rcnt_il, total_bal_il, il_util, open_rv_12m, open_rv_24m. As the objective is to build a model to predict a borrower is default or not to support making decision on loan approval. Hence, these fields are not available at at the time of loan application and should be excluded from the train features")
	
		# Select valid data
		st.markdown('<p style="color:Green; font-size: 25px;"> 3. Select valid data</p>', unsafe_allow_html=True)
		st.write("**Select features of interest**: It's important to specify which loan application are default at the initial state of loan application to help making decision about loan approvement. This analysis focus on the initial state of loan application, therefore only selecting features that describe the initial loan application, the remaining features were not sure that data were generated at the loan application or reflecting loan payment progress from data description. In order to avoiding leaky feagtures, they were not selected for building the model and therefore are excluded")
		st.write("**Select valid rows**: The following describes distinct values of loan status ard their counts")
		st.write(credit_risk.data.loan_status.value_counts())
		st.write("Only select loan_status with values ['Fully Paid', 'Charged Off', 'Default'], the remaining status indicates that the borrowers are paying the loan and not the final status, hence these rows are excluded from the final dataset.Rename 'loan_status' to 'Class' and Convert data in 'Class' var with 1 if the value was 'Fully Paid', 0 if value was ['Charged Off','Default']")

		st.markdown('<p style="color:Green; font-size: 25px;"> 4. Data Description</p>', unsafe_allow_html=True)
		credit_risk.describe_data()	
		
		st.markdown('<p style="color:Green; font-size: 25px;"> 5. Data Cleansing</p>', unsafe_allow_html=True)
		st.write('The following steps applied to in cleansing step: 1. convert data type to appropriate type, 2. select vars of interest, 3. select valid rows, 4. Rename columns from loan_status to Class, 5. Convert data in column Class: 1 if the value was Fully Paid or Charged Off, 0 if value was Fully Paid')
		cleaned_ds = credit_risk.clean_data(credit_risk.data)
		st.write(cleaned_ds.head(10))
		st.write(cleaned_ds.shape)
		st.write("After filtering and cleansing data, the dataset only consists of 228046 rows and 28 columns")
		st.write("Attributes include numerical, categorical and temporal data type")

		st.markdown('<p style="color:Green; font-size: 25px;"> 6. Descriptice statistics</p>', unsafe_allow_html=True)
		pd.options.display.float_format = '{:.3f}'.format
		st.write(cleaned_ds.describe(include='all').T)
		st.write("There are 228,046 instances in the dataset, the dataset is quite medium size. Notice that there are several fields having number of non-null values less than 228,046, meaning that these fields having missing data. We need to take care of this later")
		st.write("Attributes include numerical, categorical and temporal data type. Temporal data fields earliest_cr_line, issue_d need to be converted to appropriate data type")
		st.write("1. **count**: The following variables have missing data: emp_title, emp_length, title, revol_util")
		st.write("2. **unique entries**: These varibles have large number of unique values emp_title, title, zip_code")
		st.write("3. **freq**: Another piece of information we obtain is the most common category. Frequency emp_title is Teacher, home_ownership is MORTGAGE, purpose is debt_consolidation")
		st.write("4. **count, mean, std, min, max**: The count, mean, min, and max rows are self-explanatory. The std row shows the standard deviation (which measures how dispersed the values are). Mean of price is 540088, median is 450000 but max price is > 7700000, this variable is heavily right skewed")
		st.write("5. **quartile**: The 25%, 50% and 75% rows show the corresponding percentiles: a percentile indidates the value below which a given percentage of observations in a group of observations falls.")



	#=========================================== Exploratory Data Analysis =======================================
	if task_option == 'Exploratory Data Analysis':
		st.write('Exploratory data analysis, also known as EDA, is a crucial step in the lifecycle. Here, the main objective is to explore and understand the data in detail. The common tools for this are descriptive statistics, charts and visualizations to look at various data attributes, find associations and correlations and make a note of data quality problems if any')
		
		credit_risk = CreditRisk()
		credit_risk.load_dataset()
		cleaned_ds = credit_risk.clean_data(credit_risk.data)


		st.markdown('<p style="color:Green; font-size: 25px;"> 1. Examine missing values</p>', unsafe_allow_html=True)
		missing_data_df = clfu.show_missing_data(cleaned_ds)
		st.write(missing_data_df)


		st.markdown('<p style="color:Green; font-size: 25px;"> 2. Split data into train and test set</p>', unsafe_allow_html=True)
		credit_risk.train_test_split()
		df_train = credit_risk.clean_data(credit_risk.df_train)
		df_test = credit_risk.clean_data(credit_risk.df_test)
		st.write("Train set dimension:")
		st.write(df_train.shape)
		st.write("Test set dimension:")
		st.write(df_test.shape)
		st.write("Train set - first 5 rows:")
		st.write(df_train.head())
		target = credit_risk.TARGET

		st.markdown('<p style="color:Green; font-size: 25px;"> 3. Examine Target variable</p>', unsafe_allow_html=True)
		clfu.plot_target_distribution(df_train)
		st.write("This is an unbalanced or imbalanced problem. Notice that there are a lot more entries of people that fully paid off their loans than people that did not pay back. This is really common for classificaion problems that have to do with fraud or spam. That means we can expect to probably do very well in terms of accuracy but our precision and recall are going to be the true metrics that we'll have to evaluate our model. We should expect to perform that well on those metrics due to the fact that we have a very imbalanced dataset here.")

		st.markdown('<p style="color:Green; font-size: 25px;"> 4. Examine Discrete vars</p>', unsafe_allow_html=True)
		st.write('**inq_last_6mths**')
		# check unique values
		df_train['inq_last_6mths'].value_counts()
		inq_last_6mths_co = df_train[df_train['Class']==1].groupby("inq_last_6mths").count()['Class']
		inq_last_6mths_fp = df_train[df_train['Class']==0].groupby("inq_last_6mths").count()['Class']
		inq_last_6mths_ratio = inq_last_6mths_co/inq_last_6mths_fp
		fig, ax = plt.subplots(1,1,figsize=(6,4))
		inq_last_6mths_ratio.plot(kind='bar')
		buf = BytesIO()
		fig.savefig(buf, format="png")
		st.image(buf)
		st.write("The plot presents the ratio of default class vs. pay off class for the comparison. There is slight increase in the default ratio as inq_last_6mths increases, except that the ratio is much smaller for value o 8. This can be explain due to small sample size (only 17 cases compared to 87763 cased of values 0). This indicate this feature can be helpful to inform the status of target var in the predictive model. To give a better pattern of this feature, considering combining values of 7 and 8 into 1 value")    


		st.markdown('<p style="color:Green; font-size: 25px;"> 5. Examine Continous vars</p>', unsafe_allow_html=True)
		st.write("**Examine distribution**")
		clfu.plot_continuous_distribution(df_train, credit_risk.RAW_CONTINUOUS_VARS)
		st.write("It can be seen from the plot that int_rate and dti have the distributions of 2 classes are a bit different, with mean of class 1 is a bit higher than mean of class 0. This indicates these 2 features may be an important features to predict default observation")
		st.write(" Some variables such as 'annual_inc', 'delinq_2yrs', 'revol_bal', 'revol_util' are heavily skewed which makes it hard to inspect the distribution between two classes, removing these observations having outliers to have to better inspectation")

		st.markdown('<p style="color:Green; font-size: 25px;"> 6. Examine Categorical vars</p>', unsafe_allow_html=True)
		st.write("**Count unique values of each categorical var**")
		cat_feature_df = pd.DataFrame(columns=['Feature','count_distinct_value'])
		for e in credit_risk.RAW_CATEGORICAL_VARS:
		    count_distinct = len(df_train[e].unique())
		    cat_feature_df = cat_feature_df.append({'Feature':e,'count_distinct_value':count_distinct},ignore_index=True)
		st.write(cat_feature_df.sort_values('count_distinct_value',ascending = False))

		st.write("**sub_grade*: Examine the default and pay-off ratio between 2 classes")
		fig, ax = plt.subplots(1,1,figsize=(10,6))
		subgrade_order = sorted(df_train['sub_grade'].unique())
		sns.countplot(x='sub_grade', data=df_train, order=subgrade_order,palette='coolwarm',hue='Class')
		buf = BytesIO()
		fig.savefig(buf, format="png")
		st.image(buf)
		st.write('It can be seen from the plot that the higher subgrade the more default rate. Subgrade may be a good feature for predicitve model')

		st.markdown('<p style="color:Green; font-size: 25px;"> 7. Examine Temporal vars</p>', unsafe_allow_html=True)
		st.write('**issue_d_date**')
		df_train['issue_d_date'] = pd.to_datetime(df_train['issue_d'], format = '%b-%y')
		fig, ax = plt.subplots(1,1,figsize=(10,6))
		plt.plot(df_train.groupby(['issue_d_date','grade'])['loan_amnt'].sum().unstack().sort_index(), linewidth=2)
		buf = BytesIO()
		fig.savefig(buf, format="png")
		st.image(buf)		
		st.write("Lending Club seems to have increased the amount of money lent from 2013 onwards. The tendency indicates that they continue to grow. In addition, we can see that their major business comes from lending money to C and B grades.")
		st.write("A grades are the lower risk borrowers, borrowers that most likely will be able to repay their loans, as they are typically in a better financial situation. Borrowers within this grade are charged lower interest rates.")
		st.write("E, F and G grades represent the riskier borrowers. Usually borrowers in somewhat tighter financial situations, or for whom there is not sufficient financial history to make a reliable credit assessment. They are typically charged higher rates, as the business, and therefore the investors, take a higher risk when lending them money.")

		st.markdown('<p style="color:Green; font-size: 25px;"> 8. Correlation Matrix</p>', unsafe_allow_html=True)
		corr_matt = df_train[credit_risk.NUMERICAL_VARS + ['Class']].corr()
		mask = np.array(corr_matt)
		mask[np.tril_indices_from(mask)] = False
		fig, ax = plt.subplots(1,1,figsize=(9,9))
		sns.heatmap(corr_matt, mask=mask, vmax=.8, square=True, annot=True,  cmap='viridis')
		buf = BytesIO()
		fig.savefig(buf, format="png")
		st.image(buf)
		st.write("We can see various relationships between the features and obviously you would get a perfect correlation along the diagonal. You should have noticed almost perfect relationship between loan_amnt and installment")
		st.write("It's pretty much makes sense that the installments and the actual loan amount would be extremely correlated because they're essentially correlated bby some sort of internal formula that this company uses. if you loan someone out one million dollars you would expect that following some formula your payments your monthly payment instalments are going to be quite high and you'll probably use that same formula even if you loan someone a thousand dollars and then those payments will be likely much less.")

		st.markdown('<p style="color:Green; font-size: 25px;"> 9. the correlation of the numeric features to the new loan_repaid </p>', unsafe_allow_html=True)
		fig, ax = plt.subplots(1,1,figsize=(10,6))
		df_train.corr()['Class'].sort_values().drop('Class').plot(kind='bar')
		buf = BytesIO()
		fig.savefig(buf, format="png")
		st.image(buf)
		st.write("Interest rate has essentially the highest negative correlation with whether or not someones's goting to repay their loan which kind of makes sense. Maybe if you have an extremely high interest rate you're going to find it harder to pay of that loan.")


	#=========================================== Data Processing =======================================
	if task_option == 'Data Processing':
		credit_risk = CreditRisk()
		credit_risk.load_dataset()

		st.markdown('<p style="color:Green; font-size: 25px;"> 1. Split data to train set and test set</p>', unsafe_allow_html=True)
		st.write('In general, the model is trained and tested in the following way: The data is split into two parts. The first part is training set, it will be used for training model to learn data and inference parameters by minimizing error between model output and observed output, this is called "training error". The second part is used fro testing the "generalization" ability of the model, i.e., its ability to give the correct answer to a new case, this is called "generation error" or "test error"')
		credit_risk.train_test_split()
		st.write("Train set dimension:")
		st.write(credit_risk.df_train.shape)
		st.write("Test set dimension:")
		st.write(credit_risk.df_test.shape)
		st.write("The whole dataset is split with 80% for training and 20% for testing. This left 182.436 samples for train set and 45.610 for test set.")

		st.markdown('<p style="color:Green; font-size: 25px;"> 2. Display raw data</p>', unsafe_allow_html=True)
		st.write(credit_risk.df_train.head())
		st.markdown('<p style="color:Green; font-size: 25px;"> 3. Processed and final data for training model</p>', unsafe_allow_html=True)

		processed_train_df = credit_risk.data_processing_pipeline2(credit_risk.df_train, 1)
		st.write(processed_train_df.head())
		# store corpus to csv file
		#dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, file_name=cf.S3_DATA_PROCESSED_PATH + 'loan_data_pipeline2_train.csv', data=processed_train_df, type='s3')

		processed_test_df = credit_risk.data_processing_pipeline2(credit_risk.df_test, 0)
		dm.write_csv_file(bucket_name=cf.S3_DATA_PATH,  file_name=cf.S3_DATA_PROCESSED_PATH + 'loan_data_pipeline2_test.csv', data=processed_test_df, type='s3')
		st.write('done')

	#========================================== Predictive Model =======================================
	if task_option == 'Train Model':
		st.sidebar.subheader('')
		st.sidebar.subheader('')
		model_name = ['Select model',
					  'Logistic Regression',
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
			if st.sidebar.button("Select threshold and Train"):
				credit_risk = CreditRisk()
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



	#============================================= Prediction ==========================================
	if task_option == 'Make Prediction':
		threshold = st.sidebar.number_input("Select threshold",0.2, key="THRESHOLD")

		st.write("#### Input your data for prediction")
		installment = st.text_input("installment", '282.40')
		loan_amnt = st.text_input("loan_amnt", '8400')
		term = st.text_input("term", ' 36 months')
		int_rate = st.text_input("int_rate", '12.84')
		grade = st.text_input("grade", 'C')
		sub_grade = st.text_input("sub_grade", 'C2')
		emp_title = st.text_input("emp_title", 'Bank of A')
		emp_length = st.text_input("emp_length", '5 years')
		home_ownership = st.text_input("home_ownership", 'MORTGAGE')
		annual_inc = st.text_input("annual_inc", '94000.0')
		verification_status = st.text_input("verification_status", 'Verified')
		issue_d = st.text_input("issue_d", 'Jul-09')
		loan_status = st.text_input("loan_status", 'Charged Off')
		purpose = st.text_input("purpose", 'educational')
		title = st.text_input("title", 'Debt consolidation')
		zip_code = st.text_input("zip_code", '913xx')
		addr_state = st.text_input("addr_state", 'CA')
		dti = st.text_input("dti", '22.54')
		delinq_2yrs = st.text_input("delinq_2yrs", '0')
		earliest_cr_line = st.text_input("earliest_cr_line", 'Jul-98')
		inq_last_6mths = st.text_input("inq_last_6mths", '1')
		open_acc = st.text_input("open_acc", '14')
		pub_rec = st.text_input("pub_rec", '0')
		revol_bal = st.text_input("revol_bal", '65621')
		revol_util = st.text_input("revol_util", '81.5')
		total_acc = st.text_input("total_acc", '30')		
		initial_list_status = st.text_input("initial_list_status", 'f')
		application_type = st.text_input("application_type", 'INDIVIDUAL')


		# Add button Predict
		if st.button("Predict"):
			new_obj = dict({
							'installment':[installment],
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
							'initial_list_status':[initial_list_status],
							'application_type':[application_type]
							})

			new_obj = pd.DataFrame.from_dict(new_obj)
			credit_risk = CreditRisk()
			# data type conversion
			for key in credit_risk.DATA_TYPE:
				st.write(key)
				new_obj[key] = new_obj[key].astype(credit_risk.DATA_TYPE[key])
			st.write(new_obj)

			processed_new_obj = credit_risk.data_processing_pipeline2(new_obj, 0)
			st.write(processed_new_obj[credit_risk.TRAIN_VARS])
			model_file = os.path.join(cf.TRAINED_MODEL_PATH,"credit_risk_logistic_regression.pkl")
			model = joblib.load(model_file)


			default_prob = model.predict_proba(processed_new_obj[credit_risk.BEST_FEATURES])
			st.write("**Default Probability**: ", default_prob)
			default_pred = np.where(default_prob[:,1] > threshold, 1, 0)   
			st.write("**Default Prediction**: ", default_pred)
