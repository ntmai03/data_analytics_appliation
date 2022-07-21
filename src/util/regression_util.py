import sys
import os
from pathlib import Path
import boto3
import streamlit as st
from io import BytesIO

import pandas as pd
import numpy as np
# split data
from sklearn.model_selection import train_test_split

# Preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Modelling Helpers:
from sklearn.preprocessing import Normalizer, scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, ShuffleSplit, cross_validate
from sklearn import model_selection
from sklearn.model_selection import train_test_split

# statsmodels
import pylab
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels as statm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import math
from math import sqrt

# Regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,ElasticNet,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
# Evaluation metrics for Regression 
from sklearn import metrics
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.svm import SVR
import xgboost as xgb

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

from src import config as cf


##########################################################################################    
# 
##########################################################################################



##########################################################################################    
# function to find upper and lower boundaries for normally distributed variables
##########################################################################################
def find_normal_boundaries(var):
    upper_boundary = var.mean() + 3 * var.std()
    lower_boundary = var.mean() - 3 * var.std()
    
    print('upper_boundary, lower_boundary: ', upper_boundary, lower_boundary)
    print('total number of var: {}'.format(len(var)))
    print('number of data points with more than upper_boundary (right end outliers): {}'.format(
        (var > upper_boundary).sum()))
    print('number of data points with less than lower_boundary (left end outliers: {}'.format(
        (var < lower_boundary).sum()))
    print('% right end outliers: {}'.format((var > upper_boundary).sum() / len(var)))
    print('% left end outliers: {}'.format((var < lower_boundary).sum() / len(var)))
    
    return upper_boundary, lower_boundary


##########################################################################################    
# 
##########################################################################################
def find_skewed_boundaries(var, distance):
    # distance passed as an argument, give us the option 
    # to estimate 1.5 times or 3 times the IQR to calculate the boundaries
    IQR = var.quantile(0.75) - var.quantile(0.25) 
    lower_boundary = var.quantile(0.25) - (IQR * distance)
    upper_boundary = var.quantile(0.75) + (IQR * distance)
    
    print('upper_boundary, lower_boundary: ', upper_boundary, lower_boundary)
    print('total number of var: {}'.format(len(var)))
    print('number of data points with more than upper_boundary (right end outliers): {}'.format(
        (var > upper_boundary).sum()))
    print('number of data points with less than lower_boundary (left end outliers: {}'.format(
        (var < lower_boundary).sum()))
    print('% right end outliers: {}'.format((var > upper_boundary).sum() / len(var)))
    print('% left end outliers: {}'.format((var < lower_boundary).sum() / len(var)))
    
    return upper_boundary, lower_boundary



##########################################################################################    
# helper function for plotting residual plots 
##########################################################################################
def plot_residual(ax1, ax2, ax3, ax4, y_pred, y_real, line_label, title):
    ax1.scatter(y_real, y_pred, color='blue',alpha=0.6,label=line_label)
    ax1.set_ylabel('Predicted Y') 
    ax1.set_xlabel('Real Y')
    ax1.legend(loc='best')
    ax1.set_title(title)

    ax2.scatter(y_pred,y_real - y_pred, color='green',marker='x',alpha=0.6,label='Residual')
    ax2.set_xlabel('Predicted Y')
    ax2.set_ylabel('Residual')    
    ax2.axhline(y=0, color='black', linewidth=2.0, alpha=0.7, label='y=0')
    ax2.legend(loc='best')
    ax2.set_title('Residual Plot')
    
    ax3.hist(y_real - y_pred, bins=30, color='green', alpha=0.7)
    ax3.set_title('Histogram of residual values')
    
    stats.probplot(y_real - y_pred, plot=ax4)
    ax4.set_title('Probability of residual values')
    
    return ax1, ax2, ax3, ax4



##########################################################################################    
#  
##########################################################################################
def MissingPercentage(x):
    return df[x].isnull().sum()/len(df)



##########################################################################################    
# 
##########################################################################################
def split_data(X, y, test_size=0.2, random_state=0):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test    


##########################################################################################    
# helper function for plotting residual plots 
##########################################################################################
def train_test_set(df, test_size=0.2, random_state=0):
    
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    return train_set, test_set


    
##########################################################################################    
# 
##########################################################################################
def get_metrics(rsquare, y, pred):

    st.write('R-squared:', np.round(rsquare,4))
    st.write('MSE:', np.round(mean_squared_error(y, pred),4))
    st.write('RMSE:', np.round(sqrt(mean_squared_error(y, pred)),4))



##########################################################################################    
# 
##########################################################################################
def feature_importance(feature_importance, TRAIN_VARS):
    feature_importance = pd.Series(np.abs(feature_importance))
    feature_importance.index = TRAIN_VARS
    feature_importance.sort_values(inplace=True,ascending=True)

    fig, axes = plt.subplots(1,1,figsize=(8,7))
    feature_importance.plot.barh()
    plt.ylabel('Weight')
    plt.title('Feature Importance')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)



##########################################################################################    
# 
##########################################################################################
def show_missing_data(df):
    miss_val_df = pd.DataFrame(df.isnull().sum(), columns=['ColumnName'])
    miss_val_df['Percentage'] = 100 * df.isnull().sum()/len(df)
    miss_val_df.sort_values('Percentage', ascending=False)
    return miss_val_df



##########################################################################################    
# 
##########################################################################################
def train_cross_validation(model, X_train, y_train, k=10):
    cv_scores = cross_val_score(model,X_train,y_train,scoring='r2',cv=10)
    rmse = np.sqrt(-cross_val_score(model,X_train, y_train,scoring="neg_mean_squared_error", cv=k))
    st.write('R-squared:', np.round(cv_scores,2))
    st.write('Average R2 score:', np.round(np.mean(cv_scores),2))
    st.write('Average RMSE score:', np.round(np.mean(rmse),2))    



##########################################################################################    
#  
##########################################################################################
def plot_continuous_var(var, title):
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    plt.subplot(121)
    sns.distplot(var, hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 3})
    plt.title(title)
    plt.subplot(122)
    sns.boxplot(var)
    plt.title(title)
    # skewness and kurtosis
    st.write('Skewness: %f' % var.skew())
    st.write('Kurtosis: %f' % var.kurt())
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)


def analyze_continuous_var(var, target):   

    # Skewness and kurtosis
    st.write('Skewness: %f' % var.skew())
    st.write('Kurtosis: %f' % var.kurt())

    fig, axes = plt.subplots(1,4,figsize=(20,4))
    
    # histogram
    plt.subplot(141)
    sns.distplot(var, bins=30)
    plt.title('Histogram')
    
    # Q-Q plot
    plt.subplot(142)
    stats.probplot(var, dist='norm', plot=pylab)
    plt.ylabel('Quantiles')
    
    # Boxplot
    plt.subplot(143)
    sns.boxplot(x=var)
    plt.title('Boxplot')
    
    # Scatter plot
    plt.subplot(144)
    plt.scatter(var, target, s=5)
    plt.title('Scatter plot')

    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)
    

    



##########################################################################################    
#  
##########################################################################################
def plot_discrete_var(x, y , x_label, y_label):

    fig, axes = plt.subplots(1,2,figsize=(12,4))
    plt.subplot(121)
    sns.countplot(x=x)
    plt.subplot(122)
    sns.boxplot(x=x, y=y)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)



##########################################################################################    
#  
##########################################################################################
def plot_geographical_var(x1, y1, weight1, x2, y2, weight2, x_label=None, y_label=None):
    fig, axes = plt.subplots(1,2,figsize=(12,5))
    plt.subplot(121)
    sns.scatterplot(x=x1,y=y1,hue=weight1)
    plt.subplot(122)
    sns.scatterplot(x=x2,y=y2,hue=weight2)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)



##########################################################################################    
#  
##########################################################################################
def plot_temporal_var(var):
    fig, axes = plt.subplots(1,1,figsize=(5,3))
    var.plot()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)



##########################################################################################    
# 
##########################################################################################
def plot_correlation(df, target_var):

    corr_matt = df.corr()

    st.write("**Correlations between independent vars to check multicollinearity**")
    mask = np.array(corr_matt)
    mask[np.tril_indices_from(mask)] = False
    fig, axes = plt.subplots(1,1,figsize=(15,15))
    sns.heatmap(corr_matt, mask=mask, vmax=.8, square=True, annot=True,  cmap='viridis')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)

    st.write("**Correlations between independent vars and dependent vars**")
    fig, axes = plt.subplots(1,1,figsize=(12,5))
    corr_matt[target_var].sort_values().drop(target_var).plot(kind='bar')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)    







    
