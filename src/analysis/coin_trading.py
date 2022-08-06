#from prophet.forecaster import Prophet
import yfinance as yf
from binance import ThreadedWebsocketManager
from datetime import datetime, timedelta
import time
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import config as cf
from scipy.optimize import brute
from io import BytesIO
# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as pta
import streamlit as st

import config as cf
import util.data_manager as dm


from binance.client import Client

client = Client(api_key = cf.binance_api_key, api_secret = cf.binance_secret_key, tld = "com")

class Coin_Trading:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists. 
        
    """
    def __init__(self, symbol):
       
        self.symbol = symbol


    def get_earliest_valid_timestamp(self, interval):
        timestamp = client._get_earliest_valid_timestamp(symbol = self.symbol, interval = interval)
        start_time = pd.to_datetime(timestamp, unit = "ms")
        return start_time


    def get_historical_bitcoin_data(self, interval, start, end=None, most_recent_obs=0):

        start = str(start)
        end = str(end)

        if(most_recent_obs > 0):
            n_days = most_recent_obs * (0.5/24)
            most_recent_start = pd.to_datetime(start) - timedelta(days= n_days)
            start = str(most_recent_start)

        bars = client.get_historical_klines(symbol = self.symbol, interval = interval,
                                            start_str = start, end_str = end, limit = 1000)
        data = pd.DataFrame(bars)
        data["Date"] = pd.to_datetime(data.iloc[:,0], unit = "ms")
        data.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        try:
            assert len(data) > 0
        except AssertionError:
            print("Cannot fetch data, check spelling or time window")
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            data[column] = pd.to_numeric(data[column], errors = "coerce")
        data.set_index("Date", drop = True)
        data = data.rename(columns = {'Close': 'price'})
        self.data = data
        self.start = start
        self.end = end
        

    def plot_timeserie_data(self):
        #plot data
        fig = go.Figure()

        """
        Plot time-serie line chart of closing price on a given plotly.graph_objects.Figure object
        """
        fig = fig.add_trace(
            go.Scatter(
                x=self.data.Date,
                y=self.data.price,
                mode="lines",
                name=self.symbol,
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



    def explore_data(self, sma1, sma2, sma3):
        data = self.calculate_sma(sma1, sma2, sma3)
        self.plot_sma(data)



    def calculate_sma(self, sma1, sma2, sma3):
        data = self.data
        data['sma1'] = data['price'].rolling(sma1).mean()
        data['sma2'] = data['price'].rolling(sma2).mean()
        data['sma3'] = data['price'].rolling(sma3).mean()

        return data

    def plot_sma(self, data):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.data["Date"],
                y=self.data["price"],
                mode="lines",
                name="close price",
                line_color="yellow"
            )
        )


        fig.add_trace(
            go.Scatter(
                x=self.data["Date"],
                y=self.data["sma1"],
                mode="lines",
                name="sma1",
                line_color="red"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.data["Date"],
                y=self.data["sma2"],
                mode="lines",
                name="sma2",
                line_color="blue"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.data["Date"],
                y=self.data["sma3"],
                mode="lines",
                name="sma3",
                line_color="green"
            )
        )


        fig.update_layout(
            width=1300,
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

        st.write(fig)        




    def rsi_back_testing(self, rsi_period, sma_period, lower_th, upper_th, rsi_limit1=None,rsi_limit2=None,
                         current_position=0, cutloss_flag=0, cutloss_th=0.3, increase_flag=0,units=1):
        data = self.data.copy()
        
        data['sma'] = data['price'].rolling(sma_period).mean()
        data['rsi'] = pta.rsi(data['sma'], length = rsi_period)
        data['rsi_indicator'] = np.nan
        data['rsi_signal'] = np.nan
        data['position'] = current_position
        data['strategy'] = np.nan
        cut_loss = 0

        data['rsi_1'] = data['rsi'].shift(1)
        data['rsi_ratio'] = data['rsi']/data['rsi_1']

        data.loc[(data['rsi'] > upper_th), 'rsi_indicator'] = 0
        data.loc[data['rsi'] < lower_th, 'rsi_indicator'] = 1

        if(rsi_limit2 == None):
            data.loc[(data['rsi'] > upper_th) , 'rsi_signal'] = 'SELL'
        else:
            data.loc[(data['rsi'] > upper_th) & (data['rsi_ratio'] < rsi_limit2), 'rsi_signal'] = 'SELL'

        if(rsi_limit1 == None):
            data.loc[(data['rsi'] < lower_th) , 'rsi_signal'] = 'BUY'
        else:
            data.loc[(data['rsi'] < lower_th) & (data['rsi_ratio'] > rsi_limit1), 'rsi_signal'] = 'BUY'

        if((cutloss_flag == 0) & (increase_flag == 0)):
            for index, row in data.iterrows():
                current_price = data.loc[index, 'price']
                if(data.loc[index,'rsi_signal'] == "BUY"):
                    if(current_position == 0):
                        current_position = 1
                        data.loc[index, 'strategy'] = 'BUY'
                elif(data.loc[index,'rsi_signal'] == 'SELL'):
                    if(current_position == 1):
                        current_position = 0
                        data.loc[index, 'strategy'] = 'SELL'   
                data.loc[index,'position'] = current_position    
                
        if((cutloss_flag == 1) & (increase_flag == 0)):
            for index, row in data.iterrows():
                if(cut_loss == 0):
                    current_price = data.loc[index, 'price']
                    if(data.loc[index,'rsi_signal'] == "BUY"):
                        if(current_position == 0):
                            current_position = 1
                            data.loc[index, 'strategy'] = 'BUY'
                            buy_price = data.loc[index, 'price']
                    elif(data.loc[index,'rsi_signal'] == 'SELL'):
                        if(current_position == 1):
                            current_position = 0
                            data.loc[index, 'strategy'] = 'SELL'
                    if(current_position == 1):
                        #if(current_price < buy_price - cutloss_th):
                        difference = current_price - buy_price
                        if( difference <= - ((cutloss_th * buy_price) /100)):
                            current_position = 0
                            data.loc[index, 'strategy'] = 'SELL'   
                            cut_loss = 1
                elif(cut_loss == 1):
                    if(data.loc[index,'rsi'] > 40):
                        current_position = 1
                        data.loc[index, 'strategy'] = 'BUY'
                        buy_price = data.loc[index, 'price']    
                        cut_loss = 0
                data.loc[index,'position'] = current_position
            
        if((cutloss_flag == 1) & (increase_flag == 1)):     
            for index, row in data.iterrows():
                if(cut_loss == 0):
                    current_price = data.loc[index, 'price']
                    if(data.loc[index,'rsi_signal'] == "BUY"):
                        if(current_position == 0):
                            current_position = 1
                            data.loc[index, 'strategy'] = 'BUY'
                            buy_price = data.loc[index, 'price']
                    elif(data.loc[index,'rsi_signal'] == 'SELL'):
                        if(current_position == 1):
                            current_position = 0
                            data.loc[index, 'strategy'] = 'SELL'
                    # cut loss
                    if(current_position == 1):
                        #if(current_price < buy_price - cutloss_th):
                        difference = current_price - buy_price
                        if( difference <= - ((cutloss_th * buy_price) /100)):
                            current_position = 0
                            data.loc[index, 'strategy'] = 'SELL'   
                            cut_loss = 1
                    # increase trend => buy again (dont wait until price goes down)
                    if(current_position == 0):
                        if((data.loc[index,'rsi'] > upper_th) & ((data.loc[index,'rsi_ratio'] > rsi_limit1))):
                            current_position = 1
                            data.loc[index, 'strategy'] = 'BUY'
                            buy_price = data.loc[index, 'price']                        
                elif(cut_loss == 1):
                    if(data.loc[index,'rsi'] > 40):
                        current_position = 1
                        data.loc[index, 'strategy'] = 'BUY'
                        buy_price = data.loc[index, 'price']    
                        cut_loss = 0
                
                data.loc[index,'position'] = current_position

        self.trading_data = data


    def plot_position(self):

        data = self.trading_data
        
        fig, ax = plt.subplots(1,1,figsize=(18,8))
        ax.plot()
        plt.plot(data['price'], linewidth=2, label='price')
        ax2=ax.twinx()
        ax2.plot(data.position,color="red")
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
        

    @staticmethod
    def plot_RSI_signal(data, symbol, flag=1):
        #plot result 
        fig, ax = plt.subplots(figsize=(18, 8))
        graph = sns.lineplot(data = data, x = 'Date', y = 'price', color='blue')
        sns.lineplot(data = data, x = 'Date', y = 'sma', color='red')
        graph.set_title("SMA - RSI:" + symbol)  
        
    
        # RSI rsi_indicator
        rsi_indicator = data.loc[data['rsi_indicator'].isin([0,1])]
        rsi_indicator = rsi_indicator.reset_index(drop = True)
        
        # RSI signals
        rsi_signals = data.loc[data['rsi_signal'].isin(['BUY','SELL'])]
        rsi_signals = rsi_signals.reset_index(drop = True)
        
        # RSI strategy
        strategy = data.loc[data['strategy'].isin(['BUY','SELL'])]
        strategy = strategy.reset_index(drop = True)
        
        
        #Add rsi indicators on price 
        for x,y,z in zip(rsi_indicator['Date'], rsi_indicator['sma'], np.round(rsi_indicator['rsi'],2)):
            label = z #Label corresponds to labels in dataset
            plt.annotate(label, #text to be displayed
                        (x,y), #point for the specific label
                        textcoords="offset points", #positioning of the text
                        xytext=(0,6), #distance from text to points
                        ha='center',
                        fontsize = 13) #horizontal alignment
            
        
        # Add rsi_ratio on price 
        for x,y,z in zip(rsi_indicator['Date'], rsi_indicator['sma'], np.round(rsi_indicator['rsi_ratio'],2)):
            label = z #Label corresponds to labels in dataset
            plt.annotate(label, #text to be displayed
                            (x,y), #point for the specific label
                            textcoords="offset points", #positioning of the text
                            xytext=(0,-18), #distance from text to points
                            ha='center',
                            fontsize = 11,
                            color='green') #horizontal alignment

        if(flag == 0):
            #Add rsi_singal 
            for x,y,z in zip(rsi_signals['Date'], rsi_signals['sma'], rsi_signals['rsi_signal']):
                label = z #Label corresponds to labels in dataset
                plt.annotate(label, #text to be displayed
                            (x,y), #point for the specific label
                            textcoords="offset points", #positioning of the text
                            xytext=(0,15), #distance from text to points
                            ha='center',
                            fontsize = 15,
                            color='red') #horizontal alignment
        else:
            #Add rsi_strategy 
            for x,y,z in zip(strategy['Date'], strategy['sma'], strategy['strategy']):
                label = z #Label corresponds to labels in dataset
                plt.annotate(label, #text to be displayed
                            (x,y), #point for the specific label
                            textcoords="offset points", #positioning of the text
                            xytext=(0,18), #distance from text to points
                            ha='center',
                            fontsize = 15,
                            color='red') #horizontal alignment
        plt.legend(['Close Price', 'sma'])
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)


    def calculate_performance(self):

        self.trading_data["returns"] = np.log(self.trading_data.price.div(self.trading_data.price.shift(1)))
        self.trading_data["creturns"] = self.trading_data["returns"].cumsum().apply(np.exp)
        
        self.trading_data["strategy_returns"] =np.log(self.trading_data.price.div(self.trading_data.price.shift(1)))
        # case 1: udpate rsi returns on records not holding asset
        self.trading_data.loc[(self.trading_data['position'] == 0) & (~self.trading_data['strategy'].isin(['BUY','SELL'])), 'strategy_returns'] = 0
        # case 2: update rsi returns on records not buying to 0
        self.trading_data.loc[(self.trading_data['position'] == 1) & (self.trading_data['strategy']=='BUY'), 'strategy_returns'] = 0           
        self.trading_data["strategy_creturns"] = self.trading_data["strategy_returns"].cumsum().apply(np.exp)  


    def plot_test_performance(self):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.trading_data.Date,
                y=self.trading_data["creturns"],
                mode="lines",
                name="Buy and Hold Strategy",
                line_color="green"
            )
        )


        fig.add_trace(
            go.Scatter(
                x=self.trading_data.Date,
                y=self.trading_data["strategy_creturns"],
                mode="lines",
                name="Strategy",
                line_color="orange"
            )
        )

        fig.update_layout(
            width=1300,
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

        st.write(fig)


    def rsi_strategy(self, rsi_period, sma_period, lower_th, upper_th, rsi_limit1=None,rsi_limit2=None,
                     current_position=0, cutloss_flag=1, cutloss_th=1, increase_flag=1,units=1):
           
        data = self.data.copy()
        
        data['sma'] = data['price'].rolling(sma_period).mean()
        data['rsi'] = pta.rsi(data['sma'], length = rsi_period)
        data['rsi_indicator'] = np.nan
        data['rsi_signal'] = np.nan
        data['position'] = current_position
        data['strategy'] = np.nan
        cut_loss = 0
        buy_price = 0

        data['rsi_1'] = data['rsi'].shift(1)
        data['rsi_ratio'] = data['rsi']/data['rsi_1']
        data.loc[(data['rsi'] > upper_th), 'rsi_indicator'] = 0
        data.loc[data['rsi'] < lower_th, 'rsi_indicator'] = 1

        if(rsi_limit2 == None):
            data.loc[(data['rsi'] > upper_th) , 'rsi_signal'] = 'SELL'
        else:
            data.loc[(data['rsi'] > upper_th) & (data['rsi_ratio'] < rsi_limit2), 'rsi_signal'] = 'SELL'

        if(rsi_limit1 == None):
            data.loc[(data['rsi'] < lower_th) , 'rsi_signal'] = 'BUY'
        else:
            data.loc[(data['rsi'] < lower_th) & (data['rsi_ratio'] > rsi_limit1), 'rsi_signal'] = 'BUY'

        index = data.index[-1]
        if((cutloss_flag == 0) & (increase_flag == 0)):
            current_price = data.loc[index, 'price']
            if(data.loc[index,'rsi_signal'] == "BUY"):
                if(current_position == 0):
                    current_position = 1
                    data.loc[index, 'strategy'] = 'BUY'
            elif(data.loc[index,'rsi_signal'] == 'SELL'):
                if(current_position == 1):
                    current_position = 0
                    data.loc[index, 'strategy'] = 'SELL'   
            data.loc[index,'position'] = current_position                
            
        if((cutloss_flag == 1) & (increase_flag == 0)):
            if(cut_loss == 0):
                current_price = data.loc[index, 'price']
                if(data.loc[index,'rsi_signal'] == "BUY"):
                    if(current_position == 0):
                        current_position = 1
                        data.loc[index, 'strategy'] = 'BUY'
                        buy_price = data.loc[index, 'price']
                elif(data.loc[index,'rsi_signal'] == 'SELL'):
                    if(current_position == 1):
                        current_position = 0
                        data.loc[index, 'strategy'] = 'SELL'
                if(current_position == 1):
                    #if(current_price < buy_price - cutloss_th):
                    difference = current_price - buy_price
                    if( difference <= - ((cutloss_th * buy_price) /100)):
                        current_position = 0
                        data.loc[index, 'strategy'] = 'SELL'   
                        cut_loss = 1
            elif(cut_loss == 1):
                if(data.loc[index,'rsi'] > 40):
                    current_position = 1
                    data.loc[index, 'strategy'] = 'BUY'
                    buy_price = data.loc[index, 'price']    
                    cut_loss = 0
            data.loc[index,'position'] = current_position
            
        if((cutloss_flag == 1) & (increase_flag == 1)):     
            if(cut_loss == 0):
                current_price = data.loc[index, 'price']
                if(data.loc[index,'rsi_signal'] == "BUY"):
                    if(current_position == 0):
                        current_position = 1
                        data.loc[index, 'strategy'] = 'BUY'
                        buy_price = data.loc[index, 'price']
                elif(data.loc[index,'rsi_signal'] == 'SELL'):
                    if(current_position == 1):
                        current_position = 0
                        data.loc[index, 'strategy'] = 'SELL'
                # cut loss
                if(current_position == 1):
                    #if(current_price < buy_price - cutloss_th):
                    difference = current_price - buy_price
                    if( difference <= - ((cutloss_th * buy_price) /100)):
                        current_position = 0
                        data.loc[index, 'strategy'] = 'SELL'   
                        cut_loss = 1
                # increase trend
                if(current_position == 0):
                    if((data.loc[index,'rsi'] > upper_th) & ((data.loc[index,'rsi_ratio'] > rsi_limit1))):
                        current_position = 1
                        data.loc[index, 'strategy'] = 'BUY'
                        buy_price = data.loc[index, 'price']                        
            elif(cut_loss == 1):
                if(data.loc[index,'rsi'] > 40):
                    current_position = 1
                    data.loc[index, 'strategy'] = 'BUY'
                    buy_price = data.loc[index, 'price']    
                    cut_loss = 0
            data.loc[index,'position'] = current_position

        return data.tail(1)   


    def rsi_trading(self, start_time, bar_length, rsi_period, sma_period, lower_th, upper_th, rsi_limit1,rsi_limit2,
        position=0, cutloss_flag=1, cutloss_th=1, increase_flag=1,units=2):      
        result_df = pd.DataFrame()
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        current_time = pd.to_datetime(current_time)
        st.write('Current time: ', current_time)

        self.get_historical_bitcoin_data(interval = bar_length, start = str(start_time), end = str(current_time), most_recent_obs=0)
        index = self.data.index[-1] 
        latest_time = pd.to_datetime(self.data.loc[index, 'Date'])
        st.write('Latest time of 15mins interval: ', latest_time)  

        next_time = latest_time + timedelta(hours = 0.25)
        st.write('Waiting until ', pd.to_datetime(next_time))
        st.write('Start counting...')

        while True:                  
            if (current_time >= next_time):
                time.sleep(20)
                self.get_historical_bitcoin_data(interval = bar_length, start = str(start_time), end = str(current_time), most_recent_obs=0)
                index = self.data.index[-1]
                result = result = self.rsi_strategy(rsi_period, sma_period, lower_th, upper_th, rsi_limit1=rsi_limit1,rsi_limit2=rsi_limit2,
                    current_position=position, cutloss_flag=cutloss_flag, cutloss_th=cutloss_th, increase_flag=increase_flag,units=units)
                result_df = result_df.append(result)
                st.write(result[['Date','price','rsi','rsi_1','rsi_ratio','rsi_signal','position','strategy']],'\n')
                
                if( result.strategy.values[0] != 'N/A'):
                    if result.strategy.values[0] == 'BUY': # signal to go long
                        order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = units)
                        st.write('buying done')
                    elif result.strategy.values[0] == 'SELL': # signal to go short
                        order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = units)
                        st.write('selling done')
                current_time = pd.to_datetime(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
                next_time = pd.to_datetime(self.data.loc[index, 'Date']) + timedelta(hours = 0.25)
                st.write('waiting until ', next_time)
                st.write('start counting ...')
                # store corpus to csv file
                
                dm.write_csv_file(bucket_name=cf.S3_DATA_PATH, 
                                  file_name="/".join([cf.S3_DATA_CRYPTO_PATH, self.symbol + '_trading.csv']), 
                                  data=result_df, type='s3')


            st.write('\n', str(current_time), end=" ", flush=True)
            time.sleep(60)
            current_time = pd.to_datetime(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
