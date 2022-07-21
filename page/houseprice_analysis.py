import streamlit as st
import joblib
import sys
from pathlib import Path
import os

import numpy as np
import pandas as pd

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

from src import config as cf
from analysis.house_price import HousePrice 
from src.util import regression_util as regu

pd.options.display.float_format = '{:.3f}'.format

############################################### Main flows #######################################################
def app():
	st.sidebar.subheader('Select function')
	task_type = ['Introduction',
				 'Data Understanding',
				 'Exploratory Data Analysis',
				 'Data Processing',
				 'Predictive Model',
				 'Prediction']
	task_option = st.sidebar.selectbox('', task_type)
	st.sidebar.header('')



	#============================================= PART 1: Introduction ========================================#
	if task_option == 'Introduction':
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> Business Objective</p>', unsafe_allow_html=True)
		st.write("Real estate agents need a software tool that helps them value homes in a specific area at the push of a button. This tool needs to output a benchmark price based on the characteristics of the house such as how many rooms the home has or how much crime there is in the area plus a whole bunch of factors. Also, agents want to see the contribution of each factor in predictive model, which features of a house are more important in determining the house price and which factors are less important in determining the house price. In other words this valuation tool needs to be tractable.")
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> Defining Problem</p>', unsafe_allow_html=True)
		st.write("It is a regression problem, the analysis applies techniques like linear regression, non-linear regression to build the predictive model that allows to predict price of a given house")
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> Notebook</p>', unsafe_allow_html=True)
		st.write("More details and explanations at https://github.com/ntmai03/DataAnalysisProject/tree/main/01-Regression")


	#=========================================== PART 2: Data Understanding ====================================#
	if task_option == 'Data Understanding':

		# initialize HousePrice object for performing tasks
		houseprice = HousePrice()

		# get data from s3
		houseprice.load_dataset()
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Overview</p>', unsafe_allow_html=True)
		st.write("This phase involves taking a deep dive into the data available and understanding it in further detail before starting the process of analysis. This involves collecting the data, describing the various attributes, performing some exploratory analysis of data. This phase is important because bad data or insufficient knowledege about available data can have cascading adverse effects in the later stages of the analysis")

		# Data collection
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. Data</p>', unsafe_allow_html=True)
		st.write("This project used the house pricing dataset available on Kaggle at: https://www.kaggle.com/harlfoxem/housesalesprediction. This is historical pricing data for house sales in King County, USA and King County is essentially where Seattle is. The target variable in this dataset is the price of a particular house, information used to predict price of a particular house includes house area, number of bedrooms, number of bathrooms, and other utilities")	
		# Data snapshot
		st.write("Data snapshot of the first 10 rows")
		st.write(houseprice.data.head(10))
		st.write('The dataset has 21613 rows, 21 columns')

		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. Data Description</p>', unsafe_allow_html=True)
		houseprice.describe_data()	
	
		# Select valid data
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 4. Select valid data</p>', unsafe_allow_html=True)
		st.write("**Select features of interest**: Column id contains unique values, it is removed from the final dataset for analysis")
		st.write("**Select valid rows**: Only select rows having price > 0")
		cleaned_ds = houseprice.clean_data(houseprice.data)
		st.write(cleaned_ds.head(10))
		st.write(cleaned_ds.shape)
		st.write("After filtering data, the dataset only consists of 21613 rows and 20 columns")
		st.write("Attributes include numerical, categorical and temporal data type")
		
		# Descriptive Statistics
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 5. Descriptive statistics</p>', unsafe_allow_html=True)
		pd.options.display.float_format = '{:.3f}'.format
		st.write(cleaned_ds.describe(include='all').T)
		st.write("1. **count**: There is no missing values in this dataset")
		st.write("2. **unique entries**: For object variable, there are 372 unique values for Date attribute and 70 unique values for zipcode")
		st.write("3. **freq**: Another piece of information we obtain is the most common category. Zipcode 98103.0 is the most common value")
		st.write("4. **count, mean, std, min, max**: The count, mean, min, and max rows are self-explanatory. The std row shows the standard deviation (which measures how dispersed the values are). Mean of price is 540088, median is 450000 but max price is > 7700000, this variable is heavily right skewed")
		st.write("5. **quartile**: The 25%, 50% and 75% rows show the corresponding percentiles: a percentile indidates the value below which a given percentage of observations in a group of observations falls.")

		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 6. Examine missing values</p>', unsafe_allow_html=True)
		missing_data_df = regu.show_missing_data(cleaned_ds)
		st.write(missing_data_df)
		st.write("There is no missing data in all columns")



	#========================================= PART 3: Exploratory Data Analysis ======================================#
	if task_option == 'Exploratory Data Analysis':

		# initialize HousePrice object for performing tasks
		houseprice = HousePrice()

		# overview
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Overview</p>', unsafe_allow_html=True)
		st.write('Exploratory data analysis, also known as EDA, is a crucial step in the lifecycle. Here, the main objective is to explore and understand the data in detail. The common tools for this are descriptive statistics, charts and visualizations to look at various data attributes, find associations and correlations and make a note of data quality problems if any')
		houseprice.load_dataset()
		cleaned_ds = houseprice.clean_data(houseprice.data)

		# split data
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. Split data into train and test set</p>', unsafe_allow_html=True)
		st.write("It is best practice to set aside part of the data at this stage (before performing Exploratory Data Analysis). This is because our brain is an amazing pattern detection system, which means that it is highly prone to overfitting: if you look at the test set, you may stumple upon some seemingly interesting pattern in the test data that leads you to select a particular kind of Machine Learning model. When you estimate the generalization error using the test set, your estimate will be too optimistic and you will launch a system that will not perform as well as expected. This is called data snooping bias")
		df_train  = houseprice.clean_data(houseprice.df_train) 
		df_test  = houseprice.clean_data(houseprice.df_test) 
		st.write('Dimensionality of train set: ',df_train.shape)
		st.write('Dimensionality of test set: ',df_test.shape)
		target = houseprice.TARGET

		# categorize var types
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. Categorize vars by data type</p>', unsafe_allow_html=True)
		st.write('As depending on types of data, different charts and transformation are employed to analyze, it is useful to categorize vars by data type and have a closer look at different types of vars')
		st.write("**1. Target var**: 'price'")
		st.write("**2. Discrete vars**: 'bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade'")
		st.write("**3. Continuous vars**: 'sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15'")
		st.write("**4. Categorical vars**: 'zipcode'")
		st.write("**5.  DateTime vars**: 'date'")

		# examine target variable
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 4. Examine Target variable</p>', unsafe_allow_html=True)
		regu.plot_continuous_var(df_train[target], houseprice.TARGET)
		st.write('Notice here, it looks like most of our houses are falling somewhere between zeo and maybe around 1.5 million dollars. We could have these extreme outliers here for the really expensive houses. It may actually make sense to drop those outliers in our analysis if they are just a few points that are very extreme. And so we can essentially build a model that realistically predicts the price of a house if its intended value between 0 and 2 million dollars')
		st.write('For optimal results, we would be looking for a normal distribution of price, however has an exponential one. Obviously there are outliers in the price available. This will surely be a problem for liner regression model.')

		# examin discrete vars
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 5. Examine Discrete vars</p>', unsafe_allow_html=True)
		for var in houseprice.DISCRETE_VARS:
			st.write(var)
			regu.plot_discrete_var(df_train[var], df_train[target], var, df_train[target])
		st.write("Majority of houses have somewhere between 2 and 4 bedrooms and it looks like there's a huge mansion somewhere in this that has thirty three bedrooms; between 1 to 3 bathrooms, between 1 to 2 floors, no waterfront, no view, between 3 and 4 condition, between 6 and 8 grade.")
		st.write("In comparison with price, in general, houses with more bedrooms have higher prices. However, the house with 33 bedrooms has price lower than houses with 9 rooms. Due to lack of data for this house, we consider this house is outlier and should handle outliers in preprocessing step so that it not affect the pattern of this feature")
		st.write("For the remaining features including bathrooms, floors, waterfront, view, condtion, grade, it's quite obvious that the higher values corresponding to the higher prices. So these features may be useful in predicting house prices")

		# examine continuous vars
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 6. Examine Continous vars</p>', unsafe_allow_html=True)
		for var in houseprice.CONTINUOUS_VARS:
			st.write(var)
			regu.analyze_continuous_var(df_train[var], df_train[target])
		st.write("**1. Check normality: **")
		st.write("It can be seen from histograms that variables sqft_living, sqft_lot, sqft_above, sqft_basement, sqft_living15, sqft_lot15 are heavily skewed")
		st.write("In Q-Q plot, many of the observations lie on the 45 degree red line following the expected quantiles of the theoretical Gaussian distribution. Some observations at the lower and upper end of the value range depart from the red line, which indicates that the variables are not normally distributed")
		st.write('')
		st.write("**2. Check outliers**:")
		st.write("From the Boxplot: it can be seen that there exist outlilers in many variables: sqft_lot15, sqft_living, sqft_lot")
		st.write("Outliers are observations that lie on abnormal distance from other observations in the data, they will affect the regression dramatically in cost coefficients to be inflated as the regression will try to place the line closer to those values")
		st.write('This may make it a bit harder for some Machine Learning algorithms to detect patterns. It might be better to transform these attributes later on to have more bell-shaped distributions')	
		st.write('')
		st.write("**3. Check linear relationship**:")
		st.write("The relationship between sqft_living, sqft_living15, and price is quite linear apart from a few values around the maximal values, towards the top left side of the plot.")
		st.write("For variables such as sqft_basement, yr_built, long, lat: Here it is not so clear whether the relationship is linear.")
		st.write("The relationship is clearly not linear between sqft_lot15, sqft_lot, and price")
		st.write('')
		st.write("**4. Other notes:**")
		st.write("These attributes have very differnt scales => need to apply feature scaling before training the model")
		st.write("long and lat are geographical attribute => should be examined in other graph")
		st.write("Most of the houses were built from 1950 to 2020")
		st.write("yr_renovated: 0 means it had not been renovated. In feature engineering step, to better capture the pattern of this feature (consistent value), the value 0 should be changed to the year of building the house. Eg., house A was built in 2000 and was not renovated => yr_renovated of house A is 2000, while house B was built in 2000 but was renovated in 2005, house C was built in 2000 and renovated in 2010.")

		# explore geographical vars
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 7. Explore Geographical vars: long, lat</p>', unsafe_allow_html=True)
		long_var = houseprice.GEOGRAPHICAL_VARS[0]
		lat_var = houseprice.GEOGRAPHICAL_VARS[1]
		regu.plot_geographical_var(df_train[lat_var],df_train[long_var],df_train[target],
								   df_train.loc[df_train[target] < 1500000, lat_var], df_train.loc[df_train[target] < 1500000, long_var], df_train.loc[df_train[target] < 1500000, target],
								   'lat','long')
		st.write("The figure on the left hand side shows long, lat of house with price represented by color. Since house prices have extreme outliers, most of the house have the same color with a few dots darker")
		st.write("The figure on the right hand side has removed outliers, hence it's give a better look at areas with high prices. ""It looks like house in the center area is higher than in other areas")

		# examine categorical vars
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 8. Explore categorical vars: zipcode</p>', unsafe_allow_html=True)
		st.write("Zipcode reprsent a specific area or residential zone and is categorical data. If we have domain knowledge, we know which areas house is more expensive than other areas. To extract this information, we can apply data transfomration from categorical data to numeric data by using mean house price of each zipcode as a value corresponding to each zipcode and rank it in increasing order")
		ordered_labels = df_train.groupby(['zipcode'])['price'].mean().sort_values().index
		ordinal_label = {k:i for i, k in enumerate(ordered_labels.values, 0)} 
		df_train['zipcode_map'] = df_train['zipcode'].map(ordinal_label)
		st.write("plot zipcode after transformation from categorical to numeric data")
		regu.analyze_continuous_var(df_train['zipcode_map'], df_train[target])
		regu.plot_geographical_var(
			df_train.loc[df_train[target] < 1500000, lat_var], df_train.loc[df_train[target] < 1500000, long_var], df_train.loc[df_train[target] < 1500000, target],
			df_train.loc[df_train[target] < 1500000, lat_var], df_train.loc[df_train[target] < 1500000, long_var], df_train.loc[df_train[target] < 1500000, 'zipcode_map'],
								   'lat','long')
		st.write('Zipcode and Long,lat attributes represent the same pattern which is the geographical attributes about the area with high price and low price which makes sense, but zipcode provides the patterns clearer than long, lat')
		st.write("From the plot, it seems that prices towards the right side is more expensive")

		# examine temporal vars
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 9. Explore temporal vars: date</p>', unsafe_allow_html=True)
		df_train[houseprice.TEMPORAL_VARS] = pd.to_datetime(df_train[houseprice.TEMPORAL_VARS])
		df_train['month'] = df_train['date'].apply(lambda date:date.month)
		regu.plot_temporal_var(df_train.groupby('month').mean()[target])
		st.write('It can be seen that sale house in spring and summer (months from 3 to 6) is a bit higher than other seasons, and overall, house prices in 2015 is a bit increase compared to in 2014')

		# investigate multi-collinearity
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 10. Multivariate Analysis: relationships between vars</p>', unsafe_allow_html=True)
		# regu.plot_correlation_matrix(df_train[houseprice.NUMERICAL_VARS + [target]])
		regu.plot_correlation(df_train, target)
		st.write('It looks like sqft_living has very high correlation with the actual price of the house.')
		st.write('Features have high correlations with many other features: sqft_living, bathrooms, grade, sqft_above => The dataset has multicollinearity problem, this problem should be resolved to reduce violation in linear regression model')

		# summary
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 11. Summary</p>', unsafe_allow_html=True)
		st.write('After understanding data from Exploratory Data Analysis, the following tasks will be applied in preprocessing step to represent the patterns of data and train predictive models:')
		st.write('1. Remove vars: id')
		st.write('2. Numeric vars: Handling outliers in heavily skewed features, Transform price in linear regression model, replace yr_renovated = 0 with yr_built, scaling data')
		st.write('3. Categorical vars: Encode cat vars - calculate average price per zipcode and rank them in increasing order')
		st.write('4. Temporal vars: convert obj var to datetime type data, create new feature season from attribute date')
		st.write('5. Check and remove vars with high correlation')



	#=========================================== PART 4: Data Processing =======================================#
	if task_option == 'Data Processing':

		# introduction
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Summary</p>', unsafe_allow_html=True)
		st.write('Afer performing Exploratory Data Analysis step, the following tasks will be applied to preprocess and transform data:')
		st.write('**1. Remove unnecessary vars**: id')
		st.write("**2. Numeric vars**:") 
		st.write("+ Generate new var: sqft_ratio = sqft_living/sqft_living15. This feature represents the ratio between living's area of the house and other houses' living areas in the same neighborhood")
		st.write('**3. Categorical vars**')
		st.write("+ Encode zipcode by calculating average price per zipcode and rank them in increasing order")
		st.write('4.**Temporal vars**:')
		st.write('+ Create new feature season from attribute date, encode season by creating dummy vars')
		st.write('5.**Scaling num vars**')

		# initialize HousePrice object for performing tasks
		houseprice = HousePrice()

		# snapshot of raw data
		houseprice.load_dataset()
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. Raw data before processing</p>', unsafe_allow_html=True)
		st.write('Train set')
		st.write(houseprice.df_train.head())
		st.write('Test set')
		st.write(houseprice.df_test.head())

		# preprocessing data
		processed_X_train = houseprice.data_processing_pipeline(houseprice.X_train, train_flag=1)
		processed_X_test = houseprice.data_processing_pipeline(houseprice.X_test)

		# snapshot of preprocessed data
		st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. Processed data</p>', unsafe_allow_html=True)
		st.write('Train set')
		st.write(processed_X_train.head())
		st.write('Test set')
		st.write(processed_X_test.head())

		

	#=========================================== Predictive Model =======================================#
	if task_option == 'Predictive Model':
		st.sidebar.subheader('')
		st.sidebar.subheader('')
		model_name = ['Select model...',
					  'Linear Regression',
					  'Decision Tree',
					  'Random Forest',
					  'Gradient Boosting Tree']	

		model_option = st.sidebar.selectbox('', model_name)

		if model_option == 'Linear Regression':
			houseprice = HousePrice()
			st.markdown('<p style="color:Green; font-size: 25px;"> 1. Linear Regression Model - Baseline model</p>', unsafe_allow_html=True)
			result = houseprice.train_regression_statsmodel(flag=0)
			st.write(result.summary())
			st.write("Notice that there is strange result for coefiicients having outliers such as bedrroms, sqft_living15,.. as their coefficients are negative while it was shown in exploratory step that they have positive correlation with target var")
			st.write('--------------------------------------------------')

			st.markdown('<p style="color:Green; font-size: 25px;"> 2. Examine violations in Linear Regression assunmptions', unsafe_allow_html=True)
			st.write('**1. Examine distribution of target var**')
			regu.plot_continuous_var(pd.Series(houseprice.y_train.values), houseprice.TARGET)
			st.write('**2. Examine distribution of continuous vars**')
			for var in houseprice.OUTLIER_VARS:
				st.write(var)
				regu.analyze_continuous_var(houseprice.processed_X_train[var], houseprice.y_train)
			st.write('Variables sqft_living, sqft_lot, sqft_above, sqft_basement, sqft_living15, sqft_lot15, sqft_ratio are highly skewed which indicate they do not follow Normal distrition')
			st.write('It can be seen that there exist outlilers in many variables: bedrooms, sqft_living, sqft_lot')
			st.write('For optimal results, we would be looking for a normal distribution of price. However has an exponential one with existing outliers in the price variable. This is a problem for our regression')
			
			st.write('**3. Examine distribution of Multicolinearity**: To determine co-linearity, we use The IAF produces a measure which estimates how much larger the square root of the standard error of an estimate is compared to a situation where the variable was completely uncorrelated with the other predictors')
			vif = houseprice.check_multi_colinearity()
			st.write(vif)
			st.write('Variables have high VIF values: sqft_living, bedrooms, bathrooms, garde, sqft_living_15, sqft_ratio')
			st.write('--------------------------------------------------')

			st.markdown('<p style="color:Green; font-size: 25px;"> 3. Fixing assumption violations in Lienar Regression</p>', unsafe_allow_html=True)
			st.write('Failure to meet one or more of the model assumptions may end up in a poor model performance. In order to fix these problems, sometimes a transformation of these variables helps improve the linear regression model')
			st.write('The followings are some approaches to fix these violations: (a) Log transform target var, (b)  adjust the high-leverage points to a threshold value to alliviate the effect of them, (c) remove features with high vif value')
			result = houseprice.train_regression_statsmodel(flag=1)
			
			st.write('**1. Examine distribution of target var**')
			regu.plot_continuous_var(houseprice.y_train, houseprice.TARGET)
			st.write('**2. Examine distribution of continuous vars**')
			for var in houseprice.OUTLIER_VARS:
				st.write(var)
				regu.analyze_continuous_var(houseprice.processed_X_train[var], houseprice.y_train)
			st.write('**3. Remove features with high VIF values**')
			vif = houseprice.check_multi_colinearity()
			st.write(vif)
			st.write('**4. Feature Selection**')
			st.write('**5. Final Result**')
			st.write(result.summary())
			st.write('--------------------------------------------------')

			
		if model_option == 'Decision Tree':
			st.sidebar.markdown('max_depth')
			max_depth = st.sidebar.slider("",1, 50, 8, key="MAX_DEPTH")
			st.sidebar.markdown('min_samples_leaf')
			min_samples_leaf = st.sidebar.slider("",5, 100, 30, key="MIN_SAMPLES_LEAF")
			st.sidebar.markdown('min_samples_split')
			min_samples_split = st.sidebar.slider("",10, 100, 50, key="MIN_SAMPLES_SPLIT")
			st.sidebar.header('')
			if st.sidebar.button("Train"):
				houseprice = HousePrice()
				houseprice.decision_tree_analysis(max_depth, min_samples_leaf, min_samples_split)


		if model_option == 'Random Forest':
			st.sidebar.markdown('max_depth')
			max_depth = st.sidebar.slider("",1, 20, 8, key="MAX_DEPTH")
			st.sidebar.markdown('max_features')
			max_features = st.sidebar.slider("",5, 20, 12, key="MAX_FEATURES")
			st.sidebar.markdown('min_samples_leaf')
			min_samples_leaf = st.sidebar.slider("",50, 200, 50, key="MIN_SAMPLES_LEAF")
			st.sidebar.markdown('min_samples_split')
			min_samples_split = st.sidebar.slider("",50, 150, 100, key="MIN_SAMPLES_SPLIT")
			st.sidebar.markdown('n_estimators')
			n_estimators = st.sidebar.slider("",50, 500, 300, key="N_ESTIMATORS")
			st.sidebar.header('')
			if st.sidebar.button("Train"):
				houseprice = HousePrice()
				houseprice.random_forest_analysis(max_depth, max_features, min_samples_leaf, min_samples_split, n_estimators)


		if model_option == 'Gradient Boosting Tree':
			st.sidebar.markdown('max_depth')
			max_depth = st.sidebar.slider("",1, 20, 5, key="MAX_DEPTH")
			st.sidebar.markdown('max_features')
			max_features = st.sidebar.slider("",5, 20, 10, key="MAX_FEATURES")
			st.sidebar.markdown('min_samples_leaf')
			min_samples_leaf = st.sidebar.slider("",50, 200, 50, key="MIN_SAMPLES_LEAF")
			st.sidebar.markdown('min_samples_split')
			min_samples_split = st.sidebar.slider("",50, 150, 100, key="MIN_SAMPLES_SPLIT")
			st.sidebar.markdown('n_estimators')
			n_estimators = st.sidebar.slider("",50, 500, 300, key="N_ESTIMATORS")
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
	    	houseprice = HousePrice()
	    	new_obj = houseprice.data_processing_pipeline(new_obj)
	    	new_obj = new_obj[houseprice.TRAIN_VARS]
	    	st.write(new_obj)
	    	model_file = os.path.join(cf.TRAINED_MODEL_PATH, "house_price_gbt.pkl")
	    	model = joblib.load(model_file)
	    	st.write("**Predicted Price**: ", model.predict(new_obj))