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

from src.data_processing import diabetes_feature_engineering as fe
from src.util import data_manager as dm
from src import config as cf
# from src.pipeline import diabetes_pipeline as pl

# Evaluation metrics for Classification
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_curve, 
                             r2_score, roc_auc_score, f1_score, precision_score, recall_score, 
                             precision_recall_curve, precision_recall_fscore_support, auc, 
                             average_precision_score)



def create_confusion_matrix(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(cm)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Prediction'

    return cm_df


def plot_roc_auc(y_train, prob_train_pred, y_test, prob_test_pred):
    fpr_train, tpr_train, threshold = roc_curve(y_train, prob_train_pred[:,1])
    fpr_test, tpr_test, threshold = roc_curve(y_test, prob_test_pred[:,1])
    
    fig = plt.figure(figsize=(6,6))
    plt.title('ROC Curve Classifiers', fontsize=16)
    plt.plot(fpr_train, tpr_train, label='Train Score: {:.4f}'.format(roc_auc_score(y_train, prob_train_pred[:,1])))
    plt.plot(fpr_test, tpr_test, label='Test Score: {:.4f}'.format(roc_auc_score(y_test, prob_test_pred[:,1])))
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)',
                 xy=(0.5, 0.5), xytext=(0.6, 0.3), arrowprops=dict(facecolor='#6E726D', shrink=0.05))
    plt.legend()

    return fig 


def plot_average_precision(label, y_score):
    precision, recall, threshold = precision_recall_curve(label, y_score)
    average_precision = average_precision_score(label, y_score)
    
    fig = plt.step(recall, precision, color='#004a93', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='#48a6ff')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: \n Average Precision-Recall Score = {0:0.2f}'.format(average_precision), fontsize=16)
    
    return fig


def select_threshold(label, y_score):
    
    precision, recall, threshold = precision_recall_curve(label, y_score)

    # fig = plt.figure()
    plt.plot(threshold, precision[1:], label='Precision', linewidth=3)
    plt.plot(threshold, recall[1:], label='Recall', linewidth=3)
    plt.title('Precision and recall for different threshold values')
    plt.xlabel('Threshold')
    plt.ylabel('Precision/Recall')
    plt.legend()



class Classification:
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

    def __init__(self, X_train, y_train, X_test, y_test, TRAIN_VARS):
 
        self.TRAIN_VARS = TRAIN_VARS
        self.X_train = X_train[self.TRAIN_VARS]
        self.X_test = X_test[self.TRAIN_VARS]
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.y_train_pred = None
        self.prob_train_pred = None
        self.y_test_pred = None
        self.prob_test_pred = None
        self.model = None



    def train_logistict_regression_statsmodel(self):

        # add constant
        X_train_const = sm.add_constant(self.X_train)

        # train model
        model = sm.Logit(self.y_train, X_train_const)
        result = model.fit()

        return result


    def train_logistict_regression_sklearn(self):

        # Train model
        model = LogisticRegression(C=99999)
        model.fit(self.X_train, self.y_train)
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
        model.fit(self.X_train, self.y_train)
        self.model = model

        # default parameters
        st.write(model.get_params())


    def train_decision_tree(self):

        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
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

        return fig


    def evaluate_performance(self):

        # model prediction
        self.prediction()

        # Accuracy score
        st.write('Accurarcy Score - Train set:', accuracy_score(self.y_train, self.y_train_pred))
        st.write('Accurarcy Score - Test set:', accuracy_score(self.y_test, self.y_test_pred))
        
        # print confusion matrix
        st.write('Confusion Matrix')
        st.write(self.print_confusion_matrix())

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

   



    









