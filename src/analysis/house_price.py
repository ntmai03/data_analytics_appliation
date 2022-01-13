# Python ≥3.5 is required
import sys
from pathlib import Path
import os
from io import BytesIO

import streamlit as st

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

import statsmodels.api as sm
import sklearn
# Regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,ElasticNet,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
# Evaluation metrics for Regression 
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.svm import SVR
import xgboost as xgb
# to persist the model and the scaler
import joblib

from src.util import data_manager as dm
from src.util import regression_util as reu
from src import config as cf


# Evaluation metrics for Regression


house_price_encode_ordinal_label = os.path.join(cf.ANALYSIS_PATH, 'house_price_encode_ordinal_label.npy')
house_price_median_imputer = os.path.join(cf.ANALYSIS_PATH, 'house_price_median_imputer.npy')
house_price_knn_imputer = os.path.join(cf.ANALYSIS_PATH, 'house_price_knn_imputer.npy')
house_price_scaler = os.path.join(cf.ANALYSIS_PATH, 'house_price_scaler.pkl')
house_price_dummy_vars = os.path.join(cf.ANALYSIS_PATH, 'house_price_dummy_vars.npy')


class HousePrice:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists. 
        
    """
    params={
    'changepoint_prior_scale':0.0018298282889708827,
    'holidays_prior_scale':0.00011949782374119523,
    'seasonality_mode':'additive',
    'seasonality_prior_scale':4.240162804451275
        }

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

    TARGET = 'price'

    TEMPORAL_VARS = ['year']

    TEXT_VARS = []

    # categorical variables to encode
    #CATEGORICAL_VARS = [var for var in df.columns if df[var].dtypes == 'O' if var not in TARGET + TEXT_VARS + TEMPORAL_VARS]
    TEMP_CATEGORICAL_VARS = ['zipcode']

    CATEGORICAL_VARS = ['season']

    #NUMERICAL_VARS = [var for var in df.columns if df[var].dtypes != 'O']
    TEMP_NUMERICAL_VARS = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                      'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                      'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                      'sqft_living15', 'sqft_lot15']

    NUMERICAL_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
                      'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 
                      'sqft_basement', 'yr_built', 'yr_renovated', 'lat', 'long', 
                      'sqft_living15', 'sqft_lot15', 'sqft_ratio', 'zipcode']

    DUMMY_VARS = []

    # numerical variables with NA in train set
    NUMERICAL_VARS_WITH_NA = []

    # categorical variables with NA in train set
    CATEGORICAL_VARS_WITH_NA = []

    # variables to log transform
    NUMERICALS_LOG_VARS = []

    # drop features
    TEMPORAL_VARS = 'date'

    TRAIN_VARS = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 
                  'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
                  'yr_renovated', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sqft_ratio', 
                  'zipcode', 'season_spring', 'season_summer', 'season_winter']


    def __init__(self):
 
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_train_pred = None
        self.y_test_pred = None
        self.target = self.TARGET
        self.processed_X_train = None
        self.processed_X_test = None


    ##############################################################################################
    # Data Processing
    ##############################################################################################
    # def  impute_median_na(self, var_list, train_flag=0):


    def load_dataset(self):

        '''
        # get data from local machine
        data_file = os.path.join(cf.DATA_RAW_PATH, "diabetes.csv")
        self.data = dm.load_csv_data(data_file)
        '''

        # get data from s3
        self.data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "kc_house_data.csv")
        
        # Split data to train set and test set       
        self.X_train, self.X_test, self.y_train, self.y_test = dm.split_data(self.data, self.data[self.target])


    def prepare_dataset(self):

        # get data from s3
        df_train = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'houseprice_train.csv')
        df_test = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_PROCESSED_PATH + 'houseprice_test.csv')
      
        self.processed_X_train = df_train[self.TRAIN_VARS]
        self.y_train = df_train[self.TARGET]
        self.processed_X_test = df_test[self.TRAIN_VARS]
        self.y_test = df_test[self.TARGET]


    def clean_data(self, df):

        data = df.copy()

        # Rename columns
        data.rename(columns=self.FEATURE_MAP, inplace=True)

        # replace '?' with NA
        data.replace('?', np.NaN, inplace=True)

        # data type conversion
        for key in self.DATA_TYPE:
            data[key] = data[key].astype(self.DATA_TYPE[key])

        # Remove duplicated data
        data = data.drop_duplicates(keep = 'last')

        # Reset index
        data = data.reset_index(drop=True)

        return data


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


    def create_sqft_ratio(self, df, var1, var2):

        data = df.copy()

        data['sqft_ratio'] = data[var1]/data[var2]

        return data

    def replace_categories(self, df, var, target):

        data = df.copy()

        ordered_labels = data.groupby([var])[target].mean().sort_values().index
        ordinal_label = {k:i for i,k in enumerate(ordered_labels, 0)}

        return ordinal_label


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


    def data_processing_pipeline(self, df, train_flag=0):

        df = self.clean_data(df)
        df = self.create_season(df, self.TEMPORAL_VARS)
        df = self.create_sqft_ratio(df, 'sqft_living', 'sqft_living15')
        df = self.encode_categorical_ordinal(df, self.TEMP_CATEGORICAL_VARS, self.TARGET, train_flag)
        df = self.impute_na_median(df, self.NUMERICAL_VARS_WITH_NA, train_flag)

        data_scaled = self.scaling_data(df, self.NUMERICAL_VARS, train_flag)
        data_categorical = self.create_dummy_vars(df, self.CATEGORICAL_VARS, train_flag)
        df = pd.concat([data_scaled,data_categorical], axis=1)

        return df




    ##############################################################################################
    # Predictive Model
    ##############################################################################################
    def train_regression_statsmodel(self):

        # get train set and test set
        self.prepare_dataset()        

        # add constant
        X_train_const = sm.add_constant(self.processed_X_train)

        # train model
        model = sm.OLS(self.y_train, X_train_const)
        result = model.fit()

        return result


    def train_regression_sklearn(self):

        # get train set and test set
        self.prepare_dataset()

        # Train model
        model = LinearRegression(fit_intercept = True)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # Result Summary Table
        summary_table = pd.DataFrame(columns=['FeatureName'], data=self.TRAIN_VARS)
        summary_table['Coefficient'] = np.transpose(model.coef_)
        summary_table.index = summary_table.index + 1
        summary_table = summary_table.sort_index()

        return summary_table



    def decision_tree_analysis(self, max_depth=5, max_features=10, min_samples_leaf=50):

        # get train set and test set
        self.prepare_dataset()

        model = DecisionTreeRegressor(max_depth=max_depth, 
                                      max_features=max_features, 
                                      min_samples_leaf=min_samples_leaf)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # Model parameters
        st.markdown('#### Hyper-parameters of model')
        st.write(model.get_params())

        # Trees
        st.markdown('#### Visualize the tree')
        graph = Source(sklearn.tree.export_graphviz(
                model,
                #out_file="kchouse_tree.dot",
                out_file=None,
                feature_names=self.TRAIN_VARS,
                class_names='price',
                special_characters=False,
                rounded=True,
                filled=True,
                max_depth=3
            ))

        png_data = graph.pipe(format='png')
        with open('dtree_structure.png', 'wb') as f:
            f.write(png_data)
        st.image(png_data)


        # important features
        st.markdown('#### Feature Importance')
        reu.feature_importance(model.feature_importances_, self.TRAIN_VARS)

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

        # examine residual plot
        st.markdown('#### Assess the goodness of model fitting')
        fig, axes = plt.subplots(2,4,figsize=(12,8))
        reu.plot_residual(axes[0][0],axes[0][1],axes[0][2],axes[0][3],pred_train,self.y_train,'Decision Tree: {}'.format(train_score),'Residual plot for train data')
        reu.plot_residual(axes[1][0],axes[1][1],axes[1][2],axes[1][3],pred_test,self.y_test,'Decision Tree: {}'.format(test_score),'Residual plot for test data')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)


    def random_forest_analysis(self, max_depth=5, max_features=10, min_samples_leaf=50, n_estimators=300):

        # get train set and test set
        self.prepare_dataset()

        model = RandomForestRegressor(max_depth=max_depth, 
                                      max_features=max_features, 
                                      min_samples_leaf=min_samples_leaf,
                                      n_estimators=n_estimators)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # Model parameters
        st.markdown('#### Hyper-parameters of model')
        st.write(model.get_params())

        # important features
        st.markdown('#### Feature Importance')
        reu.feature_importance(model.feature_importances_, self.TRAIN_VARS)

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

        # examine residual plot
        st.markdown('#### Assess the goodness of model fitting')
        fig, axes = plt.subplots(2,4,figsize=(12,8))
        reu.plot_residual(axes[0][0],axes[0][1],axes[0][2],axes[0][3],pred_train,self.y_train,'Decision Tree: {}'.format(train_score),'Residual plot for train data')
        reu.plot_residual(axes[1][0],axes[1][1],axes[1][2],axes[1][3],pred_test,self.y_test,'Decision Tree: {}'.format(test_score),'Residual plot for test data')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)


    def gbt_analysis(self, max_depth=5, max_features=10, min_samples_leaf=50, n_estimators=300):

        # get train set and test set
        self.prepare_dataset()

        model = GradientBoostingRegressor(max_depth=max_depth, 
                                      max_features=max_features, 
                                      min_samples_leaf=min_samples_leaf,
                                      n_estimators=n_estimators)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # Model parameters
        st.markdown('#### Hyper-parameters of model')
        st.write(model.get_params())

        # important features
        st.markdown('#### Feature Importance')
        reu.feature_importance(model.feature_importances_, self.TRAIN_VARS)

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

        # examine residual plot
        st.markdown('#### Assess the goodness of model fitting')
        fig, axes = plt.subplots(2,4,figsize=(12,8))
        reu.plot_residual(axes[0][0],axes[0][1],axes[0][2],axes[0][3],pred_train,self.y_train,'Decision Tree: {}'.format(train_score),'Residual plot for train data')
        reu.plot_residual(axes[1][0],axes[1][1],axes[1][2],axes[1][3],pred_test,self.y_test,'Decision Tree: {}'.format(test_score),'Residual plot for test data')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)





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


    def regression_important_feature(self):
        forward_selection_features = self.forward_selection()
        st.write('Forward Selection: ')
        st.write(forward_selection_features)
        
        backward_selection_features = self.backward_elimination()
        st.write('Backward Elimination: ')
        st.write(backward_selection_features)


    def prediction(self):
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)


    
    def evaluate_performance(self):

        # model prediction
        self.prediction()

        # Accuracy score
        st.write('Accurarcy Score - Train set:', rsquared(self.y_train, self.y_train_pred))
        st.write('Accurarcy Score - Test set:', rsquared(self.y_test, self.y_test_pred))
        
    
   



    









