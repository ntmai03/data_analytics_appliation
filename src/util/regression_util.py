import sys
import os
from pathlib import Path
import boto3

import pandas as pd
# split data
from sklearn.model_selection import train_test_split

#sys.path.append('src')

import config as cf



def analyze_continuous_var(var, target):   
    plt.figure(figsize=(20,5))
    
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
    
    # Skewness and kurtosis
    print('Skewness: %f' % var.skew())
    print('Kurtosis: %f' % var.kurt())
    
    plt.show()
    
    
# function to find upper and lower boundaries for normally distributed variables

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


# helper function for plotting residual plots
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


def MissingPercentage(x):
    return df[x].isnull().sum()/len(df)



def split_data(X, y, test_size=0.2, random_state=0):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, X_test, y_train, y_test
    