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

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import config as cf


def plot_timeserie_data(x, y, title):
    #plot data
    fig = go.Figure()

    """
    Plot time-serie line chart of closing price on a given plotly.graph_objects.Figure object
    """
    fig = fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=title,
        )
    )
    fig.update_layout(
        width=1300,
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        legend=dict(x=0, y=0.99, traceorder='normal', font=dict(size=12)),
        autosize=False,
        template="plotly_dark"

    )
    st.write(fig)   

