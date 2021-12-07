# Python ≥3.5 is required
import sys
from pathlib import Path
import os

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
# for Classification models
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
#from lightgbm import LGBMClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
# to persist the model and the scaler
import joblib

from src.util import data_manager as dm
from src.util import classification_util as clf
from src import config as cf
# from src.pipeline import diabetes_pipeline as pl

# Evaluation metrics for Classification
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_curve, 
                             r2_score, roc_auc_score, f1_score, precision_score, recall_score, 
                             precision_recall_curve, precision_recall_fscore_support, auc, 
                             average_precision_score)

titanic_median_imputer = os.path.join(cf.ANALYSIS_PATH, 'titanic_median_imputer.npy')
titanic_knn_imputer = os.path.join(cf.ANALYSIS_PATH, 'titanic_knn_imputer.npy')
titanic_scaler = os.path.join(cf.ANALYSIS_PATH, 'titanic_scaler.pkl')
dummy_vars = os.path.join(cf.ANALYSIS_PATH, 'titanic_train_dummy.npy')


class Titanic:
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
    FEATURE_MAP = {'survived': 'Class',
                   'pclass': 'Pclass',
                   'sex': 'Sex',
                   'age': 'Age',
                   'sibsp': 'Sibsp',
                   'parch': 'Parch',
                   'ticket': 'Ticket',
                   'fare': 'Fare',
                   'cabin': 'Cabin',
                   'embarked': 'Embarked',}

    SELECTED_VARS = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex', 'embarked']

    # data type conversion
    DATA_TYPE = {'Class': 'int64',
                'Pclass': 'int64',
                'Sex': 'object',
                'Age': 'float64',
                'Sibsp': 'int64',
                'Parch': 'int64',
                'Ticket': 'object',
                'Fare': 'float64',
                'Cabin': 'object',
                'Embarked': 'object',}

    NUMERICAL_VARS_WITH_NA = ['Pclass', 'Age', 'Sibsp', 'Parch', 'Fare']

    NUMERICAL_VARS = ['Pclass', 'Age', 'Sibsp', 'Parch', 'Fare']

    CATEGORICAL_VARS = ['Sex', 'Embarked']

    NA_VARS = ['Age_NA', 'Cabin_NA']

    DUMMY_VARS = []

    TRAIN_VARS = ['Pclass', 'Age', 'Sibsp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']


    def __init__(self):
 
        self.TRAIN_VARS = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.y_train_pred = None
        self.prob_train_pred = None
        self.y_test_pred = None
        self.prob_test_pred = None
        self.target = 'survived'
        self.processed_X_train = None
        self.processed_X_test = None


    ##############################################################################################
    # Data Processing
    ##############################################################################################
    # def  impute_median_na(self, var_list, train_flag=0):


    def prepare_dataset(self):

        '''
        # get data from local machine
        data_file = os.path.join(cf.DATA_RAW_PATH, "diabetes.csv")
        self.data = dm.load_csv_data(data_file)
        '''

        # get data from s3
        self.data = dm.s3_load_csv(cf.S3_DATA_PATH, cf.S3_DATA_RAW_PATH + "Titanic.csv")
        
        # Split data to train set and test set       
        self.X_train, self.X_test, self.y_train, self.y_test = clf.split_data(self.data, self.data[self.target])



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


    def impute_na_median(self, df, var_list, train_flag=0):

        data = df.copy()

        if(train_flag == 1):
            median_var_dict = {}
            for var in var_list:
                median_val = data[var].median()
                median_var_dict[var] = median_val
            # save result
            np.save(titanic_median_imputer, median_var_dict)
        else:
            median_var_dict = np.load(titanic_median_imputer, allow_pickle=True).item()

        for var in var_list:
            median_var = median_var_dict[var]
            data[var].fillna(median_val, inplace=True)

        return data



    def impute_na_knn(self, df, var_list, train_flag=0):

        data = df.copy()

        imputer = IterativeImputer(n_nearest_features=None, imputation_order='ascending')

        if(train_flag == 1):
            imputer.fit(data[var_list])
            joblib.dump(imputer, titanic_knn_imputer)
        else:
            imputer = joblib.load(titanic_knn_imputer)

        data[var_list] = imputer.transform(data[var_list])

        return data


    def scaling_data(self, df, var_list, train_flag=0):

        data = df.copy()

        # fit scaler
        scaler = MinMaxScaler()
        scaler.fit(data[var_list])

        # persist the model for future use
        if(train_flag == 1):
            joblib.dump(scaler, titanic_scaler)
        scaler = joblib.load(titanic_scaler)

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
            pd.Series(train_dummy).to_csv(dummy_vars, index=False)
        else:
            test_dummy = list(data_categorical.columns)
            train_dummy = pd.read_csv(dummy_vars)
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
        df = self.impute_na_median(df, self.NUMERICAL_VARS_WITH_NA, train_flag)
        data_scaled = self.scaling_data(df, self.NUMERICAL_VARS, train_flag)
        data_categorical = self.create_dummy_vars(df, self.CATEGORICAL_VARS, train_flag)
        df = pd.concat([data_scaled,data_categorical], axis=1)

        return df










    ##############################################################################################
    # Predictive Model
    ##############################################################################################
    def train_logistict_regression_statsmodel(self):

        # add constant
        X_train_const = sm.add_constant(self.processed_X_train)

        # train model
        model = sm.Logit(self.y_train, X_train_const)
        result = model.fit()

        return result


    def train_logistict_regression_sklearn(self):

        # Train model
        model = LogisticRegression(C=99999)
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # Result Summary Table
        summary_table = pd.DataFrame(columns=['FeatureName'], data=self.TRAIN_VARS)
        summary_table['Coefficient'] = np.transpose(model.coef_)
        summary_table.index = summary_table.index + 1
        summary_table = summary_table.sort_index()
        summary_table['OddsRatio'] = np.exp(summary_table.Coefficient)
        summary_table = summary_table.sort_values('OddsRatio', ascending=False)

        return summary_table


    def train_random_forest(self):

        # Train modl
        model = RandomForestClassifier()
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # default parameters
        st.write(model.get_params())


    def train_decision_tree(self):

        model = DecisionTreeClassifier()
        model.fit(self.processed_X_train, self.y_train)
        self.model = model

        # default parameters
        st.write(model.get_params())




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


    def logistic_regression_important_feature(self):
        forward_selection_features = self.forward_selection()
        st.write('Forward Selection: ')
        st.write(forward_selection_features)
        
        backward_selection_features = self.backward_elimination()
        st.write('Backward Elimination: ')
        st.write(backward_selection_features)


    def prediction(self):
        self.y_train_pred = self.model.predict(self.X_train)
        self.prob_train_pred = self.model.predict_proba(self.X_train)

        self.y_test_pred = self.model.predict(self.X_test)
        self.prob_test_pred = self.model.predict_proba(self.X_test)


    def print_confusion_matrix(self):

        fig = plt.figure(figsize=(6,2))
        plt.subplot(1,2,1)
        sns.heatmap(create_confusion_matrix(self.y_train, self.y_train_pred), annot=True, fmt='d')
        plt.title('Train Set')
        plt.subplot(1,2,2)
        sns.heatmap(create_confusion_matrix(self.y_test, self.y_test_pred), annot=True, fmt='d')
        plt.title('Test Set')

        st.pyplot(fig)


    def evaluate_performance(self):

        # model prediction
        self.prediction()

        # Accuracy score
        st.write('Accurarcy Score - Train set:', accuracy_score(self.y_train, self.y_train_pred))
        st.write('Accurarcy Score - Test set:', accuracy_score(self.y_test, self.y_test_pred))
        
        # print confusion matrix
        st.write('Confusion Matrix')
        self.print_confusion_matrix()

        # print classification report
        st.write('classification_report')
        st.write(classification_report(self.y_train, self.y_train_pred))
        st.write(classification_report(self.y_test, self.y_test_pred))

        # plot ROC AUC
        st.write('ROC AUC')
        st.write(plot_roc_auc(self.y_train, self.prob_train_pred, self.y_test, self.prob_test_pred))

        # plot precision recall curve
        st.write('Precision - Recall curve')
        y_train_score = self.prob_train_pred[:,1]
        y_test_score = self.prob_test_pred[:,1]

        
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(1,2,1)
        plot_average_precision(self.y_train, y_train_score)
        plt.title('Train set')
        plt.subplot(1,2,2)
        plot_average_precision(self.y_test, y_test_score)
        plt.title('Test set')
        st.write(fig)
        

        # select threshold
        st.write('Select threshold')
        fig = plt.figure(figsize=(12, 5))
        plt.subplot(1,2,1)
        select_threshold(self.y_train, y_train_score)
        plt.title('Train set')
        plt.subplot(1,2,2)
        select_threshold(self.y_test, y_test_score)
        plt.title('Test set')
        st.write(fig)

   



    









