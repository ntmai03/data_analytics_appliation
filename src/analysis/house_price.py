# utitlity libraries
import sys
from pathlib import Path
import os
from io import BytesIO
import streamlit as st
import joblib

# Scikit-Learn ≥0.20 is required
import sklearn

# Dataframe manipulation
import numpy as np
import pandas as pd

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.tree import export_graphviz
from graphviz import Source
from IPython.display import Image

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

# Modelling Helpers:
from sklearn.preprocessing import Normalizer, scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, ShuffleSplit, cross_validate
from sklearn import model_selection
from sklearn.model_selection import train_test_split

# Regression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,ElasticNet,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
import xgboost as xgb

# Evaluation metrics for Regression 
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# user-defined functions
from src.util import data_manager as dm
from src.util import regression_util as reu
from src import config as cf


########################### Define paths to store calculated data #################################  
house_price_encode_ordinal_label = os.path.join(cf.ANALYSIS_PATH, cf.data['house_price_encode_ordinal_label'])
house_price_median_imputer = os.path.join(cf.ANALYSIS_PATH, cf.data['house_price_median_imputer'])
house_price_knn_imputer = os.path.join(cf.ANALYSIS_PATH, cf.data['house_price_knn_imputer'])
house_price_scaler = os.path.join(cf.ANALYSIS_PATH, cf.data['house_price_scaler'])
house_price_dummy_vars = os.path.join(cf.ANALYSIS_PATH, cf.data['house_price_dummy_vars'])



class HousePrice:

    ######################################### Define variables used in class ####################################
    # rename columns
    FEATURE_MAP = {'date': 'date',
                'price': 'price'}

    SELECTED_VARS = ['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                     'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                     'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 
                     'sqft_living15', 'sqft_lot15']

    # data type conversion
    DATA_TYPE = {'zipcode': 'str',
                 'date': 'object',
                 'price': 'float64',
                 'bedrooms': 'int64',
                 'bathrooms': 'int64',
                 'sqft_living': 'int64',
                 'sqft_lot': 'int64',
                 'floors': 'int64',
                 'waterfront': 'int64',
                 'view': 'int64',
                 'condition': 'int64',
                 'grade': 'int64',
                 'sqft_above': 'int64',
                 'sqft_basement': 'int64',
                 'yr_built': 'int64',
                 'yr_renovated': 'int64',
                 'lat': 'float64',
                 'long': 'float64',
                 'sqft_living15': 'int64',
                 'sqft_lot15': 'int64'}

    # Define variables
    TARGET = 'price'
    TEXT_VARS = []
    CATEGORICAL_VARS = ['zipcode']   
    NUMERICAL_VARS = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                      'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                      'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                      'sqft_living15', 'sqft_lot15']
    TEMPORAL_VARS = 'date'
    DISCRETE_VARS = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade']
    CONTINUOUS_VARS = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 
                        'yr_built', 'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    GEOGRAPHICAL_VARS = ['long', 'lat']


    TRAIN_NUMERICAL_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                      'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                      'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                      'sqft_living15', 'sqft_lot15', 'sqft_ratio', 'zipcode']
    TRAIN_CATEGORICAL_VARS = ['season']
    DUMMY_VARS = []
    # numerical variables with NA in train set
    NUMERICAL_VARS_WITH_NA = []
    # categorical variables with NA in train set
    CATEGORICAL_VARS_WITH_NA = []
    # variables to log transform
    NUMERICALS_LOG_VARS = []
    TRAIN_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 
                  'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
                  'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_ratio', 
                  'zipcode', 'season_spring', 'season_summer', 'season_winter']

    OUTLIER_VARS = ['sqft_lot', 'sqft_above', 'sqft_lot15','sqft_basement', 'bedrooms']
    OUTLIER_DICT = {'sqft_lot':43560, 'sqft_lot15':19647, 'sqft_above':4070.0, 'sqft_basement':1580, 'bedrooms':10}
    NO_MULTICOLINEARTITY_VARS = ['sqft_living',  'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'sqft_basement',
                                 'yr_built', 'yr_renovated', 'lat', 'long', 'zipcode', 'season_spring', 'season_summer', 'season_winter']

    def __init__(self):
 
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.target = None
        self.processed_X_train = None
        self.processed_X_test = None


    ##############################################################################################
    # Load raw data and split data to train set and test set
    ##############################################################################################
    def load_dataset(self):
        # get data from s3
        self.data = dm.read_csv_file(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + cf.data['house_price_data_file'])
        
        # Split data to train set and test set       
        self.X_train, self.X_test, self.y_train, self.y_test = reu.split_data(self.data, self.data[self.TARGET])
        self.df_train, self.df_test = reu.train_test_set(self.data)



    ##############################################################################################
    # 
    ##############################################################################################
    def load_final_dataset(self, flag = 0):
        # case 1: preprocessing data for non-linear regression models
        if(flag == 0):
            # get data from s3
            df_train = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + cf.data['houseprice_train'])
            df_test = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + cf.data['houseprice_test'])
            self.processed_X_train = df_train[self.TRAIN_VARS]
            self.y_train = df_train[self.TARGET]
            self.processed_X_test = df_test[self.TRAIN_VARS]
            self.y_test = df_test[self.TARGET]
        # case 2: fixing linear's assumption for linear regression model
        else:
            self.load_dataset()
            self.processed_X_train = self.fixing_linear_regression_violation(self.X_train)[self.TRAIN_VARS]
            self.processed_X_test = self.fixing_linear_regression_violation(self.X_test)[self.TRAIN_VARS]
            self.y_train = np.log(self.y_train)
            self.y_test = np.log(self.y_test)



    ##############################################################################################
    # 
    ##############################################################################################
    def clean_data(self, df):

        data = df.copy()

        # Rename columns
        data.rename(columns=self.FEATURE_MAP, inplace=True)

        # select columns of interest
        data = data[self.SELECTED_VARS]

        # remove invalid rows
        data = data[data[self.TARGET] > 0]

         # data type conversion
        for key in self.DATA_TYPE:
            data[key] = data[key].astype(self.DATA_TYPE[key])

        # Remove duplicated data
        data = data.drop_duplicates(keep = 'last')

        # Reset index
        data = data.reset_index(drop=True)

        return data



    ##############################################################################################
    # 
    ##############################################################################################
    def describe_data(self):
        name = ['date', 'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 
                'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
        description = [ 'Date of the home sale',
                        'Price of each home sold',
                        'Number of bedrooms',
                        'Number of bathrooms, where .5 accounts for a room with a toilet but no shower',
                        'Square footage of the apartments interior living space',
                        'Square footage of the land space',
                        'Number of floors',
                        'A dummy variable for whether the apartment was overlooking the waterfront or not',
                        'An index from 0 to 4 of how good the view of the property was',
                        'An index from 1 to 5 on the condition of the apartment',
                        'An index from 1 to 13, where 1-3 falls short of building construction and design, 7 has an average level of construction and design, and 11-13 have a high quality level of construction and design.',
                        'The square footage of the interior housing space that is above ground level',
                        'The square footage of the interior housing space that is below ground level',
                        'The year the house was initially built',
                        'The year of the house’s last renovation',
                        'What zipcode area the house is in',
                        'Lattitude',
                        'Longitude',
                        'The square footage of interior housing living space for the nearest 15 neighbors',
                        'The square footage of the land lots of the nearest 15 neighbors'
            ]
        data_describe = pd.DataFrame()
        data_describe['Name'] = name
        data_describe['Description'] = description
        st.write(data_describe)



    ##############################################################################################
    # 
    ##############################################################################################
    def create_season(self, df, var):

        data = df.copy()

        data[var] = pd.to_datetime(data[var])
        data['month'] = data[var].apply(lambda var:var.month)
        data['year'] = data[var].apply(lambda var:var.year)
        data['season'] = 'NA'
        data.loc[data.month.isin([12,1,2]), 'season'] = 'winter'
        data.loc[data.month.isin([3,4,5]), 'season'] = 'spring'
        data.loc[data.month.isin([6,7,8]), 'season'] = 'summer'
        data.loc[data.month.isin([9,10,11]), 'season'] = 'autum'

        return data



    ##############################################################################################
    # 
    ##############################################################################################
    def create_sqft_ratio(self, df, var1, var2):

        data = df.copy()

        data['sqft_ratio'] = data[var1]/data[var2]

        return data



    ##############################################################################################
    # 
    ##############################################################################################
    def replace_categories(self, df, var, target):

        data = df.copy()

        ordered_labels = data.groupby([var])[target].mean().sort_values().index
        ordinal_label = {k:i for i,k in enumerate(ordered_labels, 0)}

        return ordinal_label



    ##############################################################################################
    # 
    ##############################################################################################
    def encode_categorical_ordinal(self, df, var_list, target, train_flag=0):

        data = df.copy()

        if(train_flag == 1):
            ordinal_label_dict = {}
            for var in var_list:
                ordinal_label = self.replace_categories(data, var, target)
                ordinal_label_dict[var]= ordinal_label
            # save the dictionary
            np.save(house_price_encode_ordinal_label, ordinal_label_dict)
        else:
            ordinal_label_dict = np.load(house_price_encode_ordinal_label, allow_pickle=True).item()

        for var in var_list:
            ordinal_label = ordinal_label_dict[var]
            data[var] = data[var].map(ordinal_label)

        return data



    ##############################################################################################
    # 
    ##############################################################################################
    def impute_na_median(self, df, var_list, train_flag=0):

        data = df.copy()

        if(train_flag == 1):
            median_var_dict = {}
            for var in var_list:
                median_val = data[var].median()
                median_var_dict[var] = median_val
            # save result
            np.save(house_price_median_imputer, median_var_dict)
        else:
            median_var_dict = np.load(house_price_median_imputer, allow_pickle=True).item()

        for var in var_list:
            median_var = median_var_dict[var]
            data[var].fillna(median_val, inplace=True)

        return data




    ##############################################################################################
    # 
    ##############################################################################################
    def impute_na_knn(self, df, var_list, train_flag=0):

        data = df.copy()

        imputer = IterativeImputer(n_nearest_features=None, imputation_order='ascending')

        if(train_flag == 1):
            imputer.fit(data[var_list])
            joblib.dump(imputer, house_price_knn_imputer)
        else:
            imputer = joblib.load(house_price_knn_imputer)

        data[var_list] = imputer.transform(data[var_list])

        return data



    ##############################################################################################
    # 
    ##############################################################################################
    def scaling_data(self, df, var_list, train_flag=0):

        data = df.copy()

        # fit scaler
        scaler = MinMaxScaler()
        scaler.fit(data[var_list])

        # persist the model for future use
        if(train_flag == 1):
            joblib.dump(scaler, house_price_scaler)
        scaler = joblib.load(house_price_scaler)

        data = pd.DataFrame(scaler.transform(data[var_list]), columns=var_list)

        return data



    ##############################################################################################
    # 
    ##############################################################################################
    def create_dummy_vars(self, df, var_list, train_flag=0):  
        
        data = df.copy()
        data_categorical = pd.DataFrame()
        for var in var_list:
            data_dummies = pd.get_dummies(data[var], prefix=var, prefix_sep='_',drop_first=True)  
            data_categorical = pd.concat([data_categorical, data_dummies], axis=1)    
        
        if(train_flag == 1):
            train_dummy = list(data_categorical.columns)
            pd.Series(train_dummy).to_csv(house_price_dummy_vars, index=False)
        else:
            test_dummy = list(data_categorical.columns)
            train_dummy = pd.read_csv(house_price_dummy_vars)
            train_dummy.columns = ['Name']
            train_dummy = list(train_dummy.Name.values)   
            
        for col in train_dummy:
            if col not in data_categorical:
                data_categorical[col] = 0
        if(len(self.DUMMY_VARS) > 0):
            data_categorical = data_categorical[self.DUMMY_VARS] 
        
        return data_categorical




    ##############################################################################################
    # 
    ##############################################################################################
    def data_processing_pipeline(self, df, train_flag=0):

        df = self.clean_data(df)
        df = self.create_season(df, self.TEMPORAL_VARS)
        df = self.create_sqft_ratio(df, 'sqft_living', 'sqft_living15')
        df = self.encode_categorical_ordinal(df, self.CATEGORICAL_VARS, self.TARGET, train_flag)
        df = self.impute_na_median(df, self.NUMERICAL_VARS_WITH_NA, train_flag)

        data_scaled = self.scaling_data(df, self.TRAIN_NUMERICAL_VARS, train_flag)
        data_categorical = self.create_dummy_vars(df, self.TRAIN_CATEGORICAL_VARS, train_flag)
        df = pd.concat([data_scaled,data_categorical], axis=1)

        return df



    ##############################################################################################
    # 
    ##############################################################################################
    def check_multi_colinearity(self):
        data = self.processed_X_train.copy()
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(data[self.TRAIN_VARS].values, i) for i in range(len(self.TRAIN_VARS))]
        vif['features'] = self.TRAIN_VARS

        return vif



    ##############################################################################################
    # 
    ##############################################################################################
    def fixing_linear_regression_violation(self, df, train_flag=0):

        df = self.clean_data(df)
        for var in self.OUTLIER_DICT:
            df.loc[df[var] >= self.OUTLIER_DICT[var], var] = self.OUTLIER_DICT[var]
        df = self.create_season(df, self.TEMPORAL_VARS)
        df = self.create_sqft_ratio(df, 'sqft_living', 'sqft_living15')
        df = self.encode_categorical_ordinal(df, self.CATEGORICAL_VARS, self.TARGET, train_flag)
        df = self.impute_na_median(df, self.NUMERICAL_VARS_WITH_NA, train_flag)

        data_scaled = self.scaling_data(df, self.TRAIN_NUMERICAL_VARS, train_flag)
        data_categorical = self.create_dummy_vars(df, self.TRAIN_CATEGORICAL_VARS, train_flag)
        df = pd.concat([data_scaled,data_categorical], axis=1)

        return df



    ##############################################################################################
    # Predictive Model
    ##############################################################################################
    def train_regression_statsmodel(self, flag = 0):


        # case fixing linear regression violation
        if(flag == 1):
            self.load_final_dataset(flag = 1)
            # add constant
            self.TRAIN_VARS = self.NO_MULTICOLINEARTITY_VARS
        # not fixing linear regression violation
        else:
            self.load_final_dataset(flag = 0)


        # train model
        X_train_const = sm.add_constant(self.processed_X_train[self.TRAIN_VARS])
        model = sm.OLS(self.y_train, X_train_const)
        result = model.fit()

        return result



    ##############################################################################################
    # 
    ##############################################################################################
    def train_regression_sklearn(self):

        # get train set and test set
        # get train set and test set
        if(flag == 1):
            self.load_final_dataset(flag == 1)
            self.TRAIN_VARS = self.NO_MULTICOLINEARTITY_VARS
        else:
            self.load_final_dataset(flag == 0)

        # Train model
        model = LinearRegression(fit_intercept = True)
        model.fit(self.processed_X_train[self.TRAIN_VARS], self.y_train)
        self.model = model

        # Result Summary Table
        summary_table = pd.DataFrame(columns=['FeatureName'], data=self.TRAIN_VARS)
        summary_table['Coefficient'] = np.transpose(model.coef_)
        summary_table.index = summary_table.index + 1
        summary_table = summary_table.sort_index()

        return summary_table



    ##############################################################################################
    # 
    ##############################################################################################
    def decision_tree_analysis(self, max_depth=8, min_samples_leaf=30, min_samples_split=50, criterion='mse'):

        # get train set and test set
        self.load_final_dataset()

        model = DecisionTreeRegressor(max_depth=max_depth, 
                                      min_samples_leaf=min_samples_leaf, 
                                      min_samples_split=min_samples_split, 
                                      criterion=criterion)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model


        # prediction
        pred_train = model.predict(self.processed_X_train)
        pred_test = model.predict(self.processed_X_test)

        # R-squared
        train_score = model.score(self.processed_X_train, self.y_train)
        test_score = model.score(self.processed_X_test, self.y_test)

        # Performance metric
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Performance metrics</p>', unsafe_allow_html=True)

        st.write('Train set')
        reu.get_metrics(train_score, self.y_train, pred_train)
        st.write('Test set')
        reu.get_metrics(test_score, self.y_test, pred_test)

        # Trees
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. Visualize the tree</p>', unsafe_allow_html=True)
        graph = Source(sklearn.tree.export_graphviz(
                model,
                out_file=None,
                feature_names=self.TRAIN_VARS,
                class_names='price',
                special_characters=False,
                rounded=True,
                filled=True,
                max_depth=3
            ))

        png_data = graph.pipe(format='png')
        with open('house_price_tree.png', 'wb') as f:
            f.write(png_data)
        st.image(png_data)

        # important features
        st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3.  Feature Importance</p>', unsafe_allow_html=True)
        reu.feature_importance(model.feature_importances_, self.TRAIN_VARS)

        # examine residual plot
        st.markdown('#### Assess the goodness of model fitting')
        fig, axes = plt.subplots(2,4,figsize=(12,8))
        reu.plot_residual(axes[0][0],axes[0][1],axes[0][2],axes[0][3],pred_train,self.y_train,'Decision Tree: {}'.format(train_score),'Residual plot for train data')
        reu.plot_residual(axes[1][0],axes[1][1],axes[1][2],axes[1][3],pred_test,self.y_test,'Decision Tree: {}'.format(test_score),'Residual plot for test data')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # cross validataion
        reu.train_cross_validation(model, self.processed_X_train, self.y_train)




    ##############################################################################################
    # 
    ##############################################################################################
    def random_forest_analysis(self, max_depth=8, max_features=12, min_samples_leaf=50, min_samples_split=100, n_estimators=300):

        # get train set and test set
        self.load_final_dataset()

        model = RandomForestRegressor(max_depth=max_depth, 
                                      max_features=max_features, 
                                      min_samples_leaf=min_samples_leaf,
                                      min_samples_split=min_samples_split,
                                      n_estimators=n_estimators)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # prediction
        pred_train = model.predict(self.processed_X_train)
        pred_test = model.predict(self.processed_X_test)

        # R-squared
        train_score = model.score(self.processed_X_train, self.y_train)
        test_score = model.score(self.processed_X_test, self.y_test)

        # Performance metric
        st.markdown('#### Performance metrics')
        st.write('Train set')
        reu.get_metrics(train_score, self.y_train, pred_train)
        st.write('Test set')
        reu.get_metrics(test_score, self.y_test, pred_test)

        # important features
        st.markdown('#### Feature Importance')
        reu.feature_importance(model.feature_importances_, self.TRAIN_VARS)

        # examine residual plot
        st.markdown('#### Assess the goodness of model fitting')
        fig, axes = plt.subplots(2,4,figsize=(12,8))
        reu.plot_residual(axes[0][0],axes[0][1],axes[0][2],axes[0][3],pred_train,self.y_train,'Decision Tree: {}'.format(train_score),'Residual plot for train data')
        reu.plot_residual(axes[1][0],axes[1][1],axes[1][2],axes[1][3],pred_test,self.y_test,'Decision Tree: {}'.format(test_score),'Residual plot for test data')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # cross validataion
        st.markdown('#### 5-fold Cross Validation')
        reu.train_cross_validation(model, self.processed_X_train, self.y_train, k =5)



    ##############################################################################################
    # 
    ##############################################################################################
    def gbt_analysis(self, max_depth=5, max_features=10, min_samples_leaf=50, min_samples_split=100, n_estimators=300):

        # get train set and test set
        self.load_final_dataset()

        model = GradientBoostingRegressor(max_depth=max_depth, 
                                      max_features=max_features, 
                                      min_samples_leaf=min_samples_leaf,
                                      min_samples_split=min_samples_split,
                                      n_estimators=n_estimators)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model
        joblib.dump(model, cf.TRAINED_MODEL_PATH + '/house_price_gbt.pkl')

        # prediction
        pred_train = model.predict(self.processed_X_train)
        pred_test = model.predict(self.processed_X_test)

        # R-squared
        train_score = model.score(self.processed_X_train, self.y_train)
        test_score = model.score(self.processed_X_test, self.y_test)

        # Performance metric
        st.markdown('#### Performance metrics')
        st.write('Train set')
        reu.get_metrics(train_score, self.y_train, pred_train)
        st.write('Test set')
        reu.get_metrics(test_score, self.y_test, pred_test)

        # important features
        st.markdown('#### Feature Importance')
        reu.feature_importance(model.feature_importances_, self.TRAIN_VARS)

        # examine residual plot
        st.markdown('#### Assess the goodness of model fitting')
        fig, axes = plt.subplots(2,4,figsize=(12,8))
        reu.plot_residual(axes[0][0],axes[0][1],axes[0][2],axes[0][3],pred_train,self.y_train,'Decision Tree: {}'.format(train_score),'Residual plot for train data')
        reu.plot_residual(axes[1][0],axes[1][1],axes[1][2],axes[1][3],pred_test,self.y_test,'Decision Tree: {}'.format(test_score),'Residual plot for test data')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)

        # cross validataion
        st.markdown('#### 5-fold Cross Validation')
        reu.train_cross_validation(model, self.processed_X_train, self.y_train, k =5)



    ##############################################################################################
    # 
    ##############################################################################################
    def forward_selection(self, significance_level=0.05):
        initial_features = self.X_train.columns.tolist()
        best_features = []
        while(len(initial_features) > 0):
            remaining_features = list(set(initial_features) - set(best_features))
            new_pval = pd.Series(index=remaining_features)
            for new_column in remaining_features:
                model = sm.Logit(self.y_train, sm.add_constant(self.X_train[best_features + [new_column]])).fit()
                new_pval[new_column] = model.pvalues[new_column]
            min_p_value = new_pval.min()
            if(min_p_value < significance_level):
                best_features.append(new_pval.idxmin())
            else:
                break

        return best_features



    ##############################################################################################
    # 
    ##############################################################################################
    def backward_elimination(self, significance_level=0.05):
        features = self.X_train.columns.tolist()
        while(len(features) > 0):
            features_with_constant = sm.add_constant(self.X_train[features])
            p_values = sm.Logit(self.y_train, features_with_constant).fit().pvalues[1:]
            max_p_value = p_values.max()
            if(max_p_value >= significance_level):
                excluded_feature = p_values.idxmax()
                features.remove(excluded_feature)
            else:
                break

        return features



    ##############################################################################################
    # 
    ##############################################################################################
    def regression_important_feature(self):
        forward_selection_features = self.forward_selection()
        st.write('Forward Selection: ')
        st.write(forward_selection_features)
        
        backward_selection_features = self.backward_elimination()
        st.write('Backward Elimination: ')
        st.write(backward_selection_features)



    ##############################################################################################
    # 
    ##############################################################################################
    def prediction(self):
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)



    ##############################################################################################
    # 
    ##############################################################################################    
    def evaluate_performance(self):

        # model prediction
        self.prediction()

        # Accuracy score
        st.write('Accurarcy Score - Train set:', rsquared(self.y_train, self.y_train_pred))
        st.write('Accurarcy Score - Test set:', rsquared(self.y_test, self.y_test_pred))
        
    
   



    









