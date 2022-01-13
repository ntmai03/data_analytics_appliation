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

# statsmodels
import pylab
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels as statm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
import math
from math import sqrt

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

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

from src.util import data_manager as dm
from src.util import classification_util as clfu
from src import config as cf

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
    
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    plt.title('ROC Curve Classifiers', fontsize=16)
    ax.plot(fpr_train, tpr_train, label='Train Score: {:.4f}'.format(roc_auc_score(y_train, prob_train_pred[:,1])))
    ax.plot(fpr_test, tpr_test, label='Test Score: {:.4f}'.format(roc_auc_score(y_test, prob_test_pred[:,1])))
    ax.plot([0,1], [0,1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)',
                 xy=(0.5, 0.5), xytext=(0.6, 0.3), arrowprops=dict(facecolor='#6E726D', shrink=0.05))
    plt.legend()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)


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



def split_data(X, y, test_size=0.2, random_state=0):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state,
                                                        stratify=y)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test
    

def get_metrics(true_labels, predicted_labels, predicted_prob):

    #st.write('Accuracy:', np.round(accuracy_score(true_labels, predicted_labels),4))
    #st.write('Precision:', np.round(precision_score(true_labels, predicted_labels,average='weighted'),4))
    #st.write('Recall:', np.round(recall_score(true_labels,predicted_labels,average='weighted'),4))
    #st.write('F1 Score:', np.round(f1_score(true_labels,predicted_labels,average='weighted'),4))
    st.write('ROC-AUC: {}'.format(np.round(roc_auc_score(true_labels, predicted_prob),4)))                   

def train_predict_model(classifier, 
                        train_features, train_labels, 
                        test_features, test_labels):
    # build model    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    return predictions    


def display_confusion_matrix(true_labels, predicted_labels, classes=[0,1]):
    
    total_classes = len(classes)
    level_labels = [total_classes*[0], list(range(total_classes))]

    cm = confusion_matrix(y_true=true_labels, y_pred=predicted_labels,labels=classes)
    cm_df = pd.DataFrame(cm)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    st.write(cm_df) 
    
def display_classification_report(true_labels, predicted_labels, classes=[0,1]):

    report = classification_report(y_true=true_labels, 
                                           y_pred=predicted_labels, 
                                           labels=classes) 
    st.write(report)
    
    
    
def display_model_performance_metrics(true_labels, predicted_labels, predicted_prob, classes=[0,1]):
    #st.write('Model Performance metrics:')
    get_metrics(true_labels=true_labels, predicted_labels=predicted_labels, predicted_prob=predicted_prob)
    st.write('\nModel Classification report:')
    display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels, classes=classes)
    st.write('\nPrediction Confusion Matrix:')
    display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels, classes=classes)


def feature_importance(feature_importance, TRAIN_VARS):
    feature_importance = pd.Series(np.abs(feature_importance))
    feature_importance.index = TRAIN_VARS
    feature_importance.sort_values(inplace=True,ascending=True)

    fig, axes = plt.subplots(1,2,figsize=(6,7))
    feature_importance.plot.barh()
    plt.ylabel('Multivariate Linear Regression')
    plt.title('Feature Importance')
    buf = BytesIO()
    fig.savefig(buf, format="png")
    st.image(buf)


def precision_recall_threshold(pred_proba, label):
    t_recall_0, t_recall_1 = [], []
    t_precision_0, t_precision_1 = [], []
    
    for threshold in np.arange(0, 1, 0.01):
        precision, recall, fscore, support = \
            precision_recall_fscore_support(label, np.where(pred_proba[:,1] > threshold, 1, 0))
        recall_0, recall_1 = recall
        precision_0, precision_1 = precision
            
        t_recall_0.append(recall_0)
        t_recall_1.append(recall_1)
        t_precision_0.append(precision_0)
        t_precision_1.append(precision_1)
    return t_precision_0, t_precision_1, t_recall_0, t_recall_1


def plot_recall_vs_decision_boundary(ax, t_recall_1,t_recall_0,threshold):
    ax.plot(np.arange(0,1,0.01), t_recall_1, label='Class1')
    ax.plot(np.arange(0,1,0.01), t_recall_0, label='Class0')
    ax.plot([.5, .5], [0,1], 'k--')
    ax.plot([threshold, threshold], [0,1], 'k--')
    plt.ylim([0.0, 1.01])
    plt.xlim([0.0, 1.01])
    ax.legend(loc='upper left', fontsize = 14)
    ax.set_title('Recall vs. Decision Boundary\n'
              'using Logistic Regression Classifier',
              fontsize=14)
    ax.set_xlabel('Decision Boundary (T)', fontsize=14)
    ax.set_ylabel('Recall Rate', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)

    return ax


def plot_precision_recall_curve(ax, label, y_score):
    
    precision, recall, threshold = precision_recall_curve(label, y_score)

    ax.plot(threshold, precision[1:], label='Precision', linewidth=3)
    ax.plot(threshold, recall[1:], label='Recall', linewidth=3)
    ax.set_title('Precision and recall for different threshold values')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Precision/Recall')
    ax.legend()