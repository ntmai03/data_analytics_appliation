#from prophet.forecaster import Prophet
import yfinance as yf
import datetime
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

#from sklearn.metrics import mean_absolute_percentage_error

class Stock:
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

    def __init__(self, symbol="GOOG"):
       
        self.end = datetime.datetime.today()
        self.start = self.end - datetime.timedelta(days=4)
        self.symbol = symbol
        self.data = self.load_data(self.start, self.end)
        self.model = None


    @st.cache(show_spinner=False)
    def load_data(self, start, end, inplace=False):
        """
        takes a start and end dates, download data do some processing and returns dataframe
        """

        data = yf.download(self.symbol, start, end + datetime.timedelta(days=1))
        try:
            assert len(data) > 0
        except AssertionError:
            print("Cannot fetch data, check spelling or time window")
        data.reset_index(inplace=True)
        data.rename(columns={"Date": "datetime"}, inplace=True)
        data["date"] = data.apply(lambda raw: raw["datetime"].date(), axis=1)

        data = data[["date", 'Close']]
        if inplace:
            self.data = data
            self.start = start
            self.end = end
            return True
        return data


    def plot_raw_data(self, fig):
        """
        Plot time-serie line chart of closing price on a given plotly.graph_objects.Figure object
        """
        fig = fig.add_trace(
            go.Scatter(
                x=self.data.date,
                y=self.data['Close'],
                mode="lines",
                name=self.symbol,
            )
        )
        return fig


    @staticmethod
    def nearest_business_day(DATE: datetime.date):
        """
        Takes a date and transform it to the nearest business day, 
        static because we would like to use it without a stock object.
        """
        if DATE.weekday() == 5:
            DATE = DATE - datetime.timedelta(days=1)

        if DATE.weekday() == 6:
            DATE = DATE + datetime.timedelta(days=1)
        return DATE


    def show_delta(self):
        """
        Visualize a summary of the stock change over the specified time period
        """
        epsilon = 1e-6
        i = self.start
        j = self.end
        s = self.data.query("date==@i")['Close'].values[0]
        e = self.data.query("date==@j")['Close'].values[0]

        difference = round(e - s, 2)
        change = round(difference / (s + epsilon) * 100, 2)
        e = round(e, 2)
        cols = st.columns(2)
        (color, marker) = ("green", "+") if difference >= 0 else ("red", "-")

        cols[0].markdown(
            f"""<p style="font-size: 90%;margin-left:5px">{self.symbol} \t {e}</p>""",
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            f"""<p style="color:{color};font-size:90%;margin-right:5px">{marker} \t {difference} {marker} {change} % </p>""",
            unsafe_allow_html=True
        ) 


    @st.cache(show_spinner=False)
    def prepare_train_test_data(self):
        """Returns two dataframes for testing and training"""

        train_data = self.load_data(st.session_state.TRAIN_START, st.session_state.TRAIN_END)
        test_data = self.load_data(st.session_state.TEST_START, st.session_state.TEST_END)
        self.train_data = train_data
        self.test_data = test_data


    @st.cache(show_spinner=False)
    def preprocess_data(self):

        X_train_value = self.train_data.Close.values.reshape(-1,1)
        X_test_value = self.test_data.Close.values.reshape(-1,1)

        # scale data
        scaler = MinMaxScaler(feature_range = (0, 1))
        train_scaled = scaler.fit_transform(X_train_value)
        dataset_total = pd.concat((self.train_data['Close'], self.test_data['Close']), axis = 0)
        inputs = dataset_total[len(dataset_total) - len(self.test_data) - 60:].values
        inputs = inputs.reshape(-1,1)
        inputs = scaler.transform(inputs)

        # creating a data structure with 60 timesteps and 1 output
        X_train = []
        y_train = []
        for i in range(60, len(self.train_data)):
            X_train.append(train_scaled[i-60:i, 0])
            y_train.append(train_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        X_test = []
        for i in range(60, len(self.test_data) + 60):
            X_test.append(inputs[i-60:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.scaler = scaler


    @st.cache(show_spinner=False)
    def train_model(self):
        # LSTM architecture
        keras.backend.clear_session()
        # initialize LSTM
        regressor = Sequential()
        # Add 1st layer
        regressor.add(LSTM(units = 100, input_shape = (self.X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))
        # Adding the output layer
        regressor.add(Dense(units = 1))
        # Compiling model
        regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
        regressor.fit(self.X_train, self.y_train, epochs = 30, batch_size = 30)
        # predict
        pred_test = regressor.predict(self.X_test)
        pred_test = pred_test.reshape(-1,1)
        self.test_data["pred_test"] = self.scaler.inverse_transform(pred_test)      


    def plot_test(self):

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.test_data["date"],
                y=self.test_data["Close"],
                mode="lines",
                name="True Closing price",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.test_data["date"],
                y=self.test_data["pred_test"],
                mode="lines",
                name="Predicted CLosing price",
                line_color="orange"
            )
        )

        '''
        fig.add_trace(
            go.Scatter(
                x=self.test_data["ds"],
                y=self.test_data["yhat_upper"],
                fill=None,
                mode="lines",
                name="CI+",
                line_color="orange",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.test_data["ds"],
                y=self.test_data["yhat_lower"],
                fill="tonexty",
                fillcolor='rgba(100,69,0,0.2)',
                mode="lines",
                line_color="orange",
                name="CI-",
            )
        )
        '''
        fig.update_layout(
            width=st.session_state.CHART_WIDTH,
            margin=dict(l=0, r=0, t=0, b=0, pad=0),
            legend=dict(
                x=0,
                y=0.99,
                traceorder="normal",
                font=dict(size=12),
            ),
            autosize=False,
            template="plotly_dark",
        )

        return fig


    def plot_inference(self) -> go.Figure:
        """
        Generate forecasts for the given horizon and plots them, returns a plotly.graph_objects.Figure
        """
        all_df=pd.concat([self.train_data,self.test_data[['ds','y']]])
        m=Prophet(**self.params)
        m.fit(all_df)
        self.model=m
        future=self.model.make_future_dataframe(periods=st.session_state.HORIZON,include_history=False)
        forecasts=self.model.predict(future)

        fig=go.Figure()
        fig.add_trace(
        go.Scatter(
            x=forecasts["ds"],
            y=forecasts["yhat"],
            mode="lines",
            name="Predicted CLosing price",
        )
        )

        fig.add_trace(
        go.Scatter(
            x=forecasts["ds"],
            y=forecasts["yhat_upper"],
            fill=None,
            mode="lines",
            name="CI+",
            line_color="orange",
        )
        )

        fig.add_trace(
        go.Scatter(
            x=forecasts["ds"],
            y=forecasts["yhat_lower"],
            fill="tonexty",
            fillcolor='rgba(100,69,0,0.2)',
            mode="lines",
            line_color="orange",
            name="CI-",
        )
        )
        fig.update_layout(
        width=st.session_state.CHART_WIDTH,
        margin=dict(l=0, r=0, t=0, b=0, pad=0),
        legend=dict(
            x=0,
            y=0.99,
            traceorder="normal",
            font=dict(size=12),
        ),
        autosize=False,
        template="plotly_dark",
        )
        return fig


    @staticmethod
    def train_test_forecast_report(symb): 
        """Launch training and plot testing results and reports MAPE error, finally it plots forecasts up to the specified horizon"""
        if st.session_state.TRAIN_JOB or st.session_state.TRAINED:
            text=st.empty() # Because streamlit adds widgets sequentially, we have to reserve a place at the top (after the chart of part 1)
            bar=st.empty() # Reserve a place for a progess bar
            
            text.write('Training model ... ') 
            bar=st.progress(0)

            # Initialize stock
            stock = Stock(symb) 
            bar.progress(10)
            
            #load train test data into the stock object, it's using cache
            stock.prepare_train_test_data() 
            bar.progress(20)

            # preprocessdata
            stock.preprocess_data()
            bar.progress(30)

            # train data
            stock.train_model()
            bar.progress(80)

            text.write('Plotting test results ...')
            fig = stock.plot_test()
            bar.progress(100)
            bar.empty() #Turn the progress bar object back to what it was before and empty container

            #st.markdown(
            #    f"## {symb} stock forecasts on testing set, Testing error {round(stock.test_mape*100,2)}%"
            #)
            st.plotly_chart(fig)
            text.write('Generating forecasts ... ')
            #fig2=stock.plot_inference() #Generate forecasts and plot them (no cache but figures are not updated if their data didn't change)
            #st.markdown(f'## Forecasts for the next {st.session_state.HORIZON} days')
            #st.plotly_chart(fig2)
            text.empty()
            """The button click will trigger this code to run only once, 
               the following flag TRAINED will keep this block of code executing even after the click,
               it won't redo everything however because we are using cache. 
               this flag needs to be initialized to False in the session state in main.py before the button"""

            st.session_state.TRAINED=True 
        else:
            st.markdown('Setup training job and hit Train')

        
    def save_forecasts(self,path):
        self.forecasts.to_csv(path)
