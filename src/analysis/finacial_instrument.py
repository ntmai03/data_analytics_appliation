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
import talib

import config as cf



from binance.client import Client

client = Client(api_key = cf.binance_api_key, api_secret = cf.binance_secret_key, tld = "com")

class FinancialInstrument:
    """
    This class enables data loading, plotting and statistical analysis of a given stock,
     upon initialization load a sample of data to check if stock exists. 
        
    """
    def __init__(self, symbol):
       
        self.symbol = symbol
        self.flag = 0


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
        data.set_index("Date", inplace = False)
        data = data.rename(columns = {'Close': 'price'})
        self.data = data
        self.start = start
        self.end = end
        

    @st.cache(show_spinner=False)
    def load_historical_stock_data(self, start, end, inplace=False):
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
                x=self.data.index,
                y=self.data['price'],
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
        s = self.data.query("index==@i")['Close'].values[0]
        e = self.data.query("index==@j")['Close'].values[0]

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


    def plot_test_performance(self):

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data["creturns"],
                mode="lines",
                name="Buy and Hold Strategy",
                line_color="orange"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data["SMA_returns"],
                mode="lines",
                name="SMA Strategy",
                line_color="blue"
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

        st.write(fig)        

    def plot_SMA(self):
        st.write(self.prepared_data.head())
        fig, axes = plt.subplots(1,1,figsize=(18,8))
        plt.plot(self.prepared_data.loc[["price","SMA_S", "SMA_L", "direction"]], fontsize = 12, secondary_y = "direction",
                                                title = "SMA{} | SMA{}".format(self.sma_s, self.sma_l))
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)





class Spot_Trading():  # Triple SMA Crossover
    
    def __init__(self, symbol, bar_length, units, position = 0):
        
        self.symbol = symbol
        self.bar_length = bar_length
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.units = units
        self.position = position
        self.cum_profits = 0 # NEW
        self.trades = 0 
        self.trade_values = []
        bar = None
        self.flag = 0
        
        #*****************add strategy-specific attributes here******************
        #self.SMA_S = sma_s
        #self.SMA_L = sma_l
        #************************************************************************
        
        
    def get_historical_data(self, symbol, interval, start, end = None, most_recent_obs=0):

        if(most_recent_obs > 0):
            n_days = most_recent_obs * (0.5/24)
            most_recent_start = pd.to_datetime(start) - timedelta(days= n_days)
            start = str(most_recent_start)

        bars = client.get_historical_klines(symbol = symbol, interval = interval,
                                            start_str = start, end_str = end, limit = 1000)

        data = pd.DataFrame(bars)
        data["Date"] = pd.to_datetime(data.iloc[:,0], unit = "ms")
        data.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            data[column] = pd.to_numeric(data[column], errors = "coerce")
        data.set_index("Date", inplace = False)
        data = data.rename(columns = {'Close': 'price'})
        self.data = data

       
    def optimize_parameters(self, SMA_S_range, SMA_L_range):
        opt = brute(self.testing_SMA, (SMA_S_range, SMA_L_range), finish=None)
        return opt  #, -self.test_strategy(opt)  
    

       
    def stream_candles(self, msg):
        
        st.write(msg)
        # extract the required items from msg
        event_time = pd.to_datetime(msg["E"], unit = "ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first   = float(msg["k"]["o"])
        high    = float(msg["k"]["h"])
        low     = float(msg["k"]["l"])
        close   = float(msg["k"]["c"])
        volume  = float(msg["k"]["v"])
        complete=       msg["k"]["x"]
        
        # print out
        st.write(".", end = "", flush = True) 
        
        # prepare features and define strategy/trading positions whenever the latest bar is complete
        if complete == True:
            # feed df (add new bar / update latest bar)
            self.data.loc[start_time] = [first, high, low, close, volume]
            self.SMA_strategy()
            # self.execute_trades()
            self.test_trading()
            
            
    def get_values(self, bar):
        date = str(self.data.index[bar].date())
        price = round(self.data.price.iloc[bar], 5)
        
        return date, price
    
            
    def buy_instrument(self, bar, units = None, amount = None):
        date, price = self.get_values(bar)
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        #self.current_balance -= units * price # reduce cash balance by "purchase price"
        #self.units += units
        self.trades += 1
        #self.trade_values.append(units * price)
        print("{} |  Buying {} for {}".format(date, units, round(price, 5)))
    
    def sell_instrument(self, bar, units = None, amount = None):
        date, price = self.get_values(bar)
        if amount is not None: # use units if units are passed, otherwise calculate units
            units = int(amount / price)
        #self.current_balance += units * price # increases cash balance by "purchase price"
        #self.units -= units
        self.trades += 1
        #self.trade_values.append(- units * price)
        print("{} |  Selling {} for {}".format(date, units, round(price, 5)))
        
    # helper method
    def go_long(self, bar, units = None, amount = None):
        if self.position == -1:
            self.buy_instrument(bar, units = -self.units) # if short position, go neutral first
        if units:
            self.buy_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.buy_instrument(bar, amount = amount) # go long

    # helper method
    def go_short(self, bar, units = None, amount = None):
        if self.position == 1:
            self.sell_instrument(bar, units = self.units) # if long position, go neutral first
        if units:
            self.sell_instrument(bar, units = units)
        elif amount:
            if amount == "all":
                amount = self.current_balance
            self.sell_instrument(bar, amount = amount) # go short
            

    def testing_SMA(self, SMA): # "strategy-specific"  

        SMA_S = SMA[0]
        SMA_L = SMA[1]       
        data = self.data.copy()
        #******************** define your strategy here ************************
        data["SMA_S"] = data.price.rolling(SMA_S).mean()
        data["SMA_L"] = data.price.rolling(SMA_L).mean()
        data["direction"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["prev_direction"] = data['direction'].shift(1)
        data["returns"] = np.log(data.price.div(data.price.shift(1)))
        data['diff'] = data['direction'] - data['prev_direction']
        data['position'] = 0
        data.loc[data['diff'] == 2, 'position'] = 1
        data.loc[data['diff'] == -2, 'position'] = -1
        data.dropna(inplace=True)
        data['strategy'] = 0
        data.loc[data.prev_direction == 1, "strategy"] = data.prev_direction * data["returns"]
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)   
        perf = data['cstrategy'][-1]

        if(self.flag == 0):
            self.prepared_data = data

        return -round(perf, 6)      
    
            
    def SMA_strategy(self): # "strategy-specific"  

        SMA_S = self.SMA_S
        SMA_L = self.SMA_L     
        data = self.data.copy()
        #******************** define your strategy here ************************
        data["SMA_S"] = data.price.rolling(SMA_S).mean()
        data["SMA_L"] = data.price.rolling(SMA_L).mean()
        # data["direction"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        cond1 = (data.SMA_S > data.SMA_L)
        cond2 = (data.SMA_S < data.SMA_L)
        data["direction"] = 0
        data.loc[cond1, "direction"] = 1
        data.loc[cond2, "direction"] = -1
        self.prepared_data = data
        self.bar = len(self.prepared_data) - 1
        
                    
    def test_trading(self): # Adj! 
        st.write('test trading')
        # nice printout
        stm = "Symbol | {} | Time | {} | current price = {} | SMA_S = {} , SMA_L = {}".format(self.symbol, self.prepared_data.index[self.bar],self.prepared_data["price"].iloc[-1],self.prepared_data["SMA_S"].iloc[-1], self.prepared_data["SMA_L"].iloc[-1])
        st.write("-" * 75)
        st.write(stm)
        st.write("-" * 75)
        
        if self.prepared_data["SMA_S"].iloc[-1] > self.prepared_data["SMA_L"].iloc[-1]: # signal to go long
            self.prepared_data.direction.iloc[-1] = 1
            if self.position in [0, -1]:
                #self.go_long(self.bar, units = self.units) # go long 
                self.buy_instrument(self.bar, units = self.units)
                # order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.position = 1  # long position
                self.report_testing_trade('GOING LONG')
        elif self.prepared_data["SMA_S"].iloc[-1] < self.prepared_data["SMA_L"].iloc[-1]: # signal to go short
            self.prepared_data.direction.iloc[-1] = -1
            if self.position in [1]:
                #self.go_short(self.bar, units = self.units) # go short with full amount
                self.sell_instrument(self.bar, units = self.units)
                self.position = -1 # short position
                self.report_testing_trade('GOING SHORT')
            #print(bar, self.data.index[bar].date(), self.data.SMA_S.iloc[bar], self.data.SMA_L.iloc[bar], self.data.direction.iloc[bar], sep = " | ")
        time.sleep(0.1) 
        
            
    def execute_trades(self): # Adj! 
        # nice printout
        stm = "Symbol | {} | Time | {} | current price = {} | SMA_S = {} , SMA_L = {}".format(self.symbol, self.prepared_data.index[self.bar],self.prepared_data["price"].iloc[-1],self.prepared_data["SMA_S"].iloc[-1], self.prepared_data["SMA_L"].iloc[-1])
        print("-" * 75)
        print(stm)
        print("-" * 75)
        
        if self.prepared_data["SMA_S"].iloc[-1] > self.prepared_data["SMA_L"].iloc[-1]: # signal to go long
            self.prepared_data.direction.iloc[-1] = 1
            if self.position in [0, -1]:
                order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                self.position = 1  # long position
                self.report_trade(order, 'GOING LONG')
        elif self.prepared_data["SMA_S"].iloc[-1] < self.prepared_data["SMA_L"].iloc[-1]: # signal to go short
            self.prepared_data.direction.iloc[-1] = -1
            if self.position in [1]:
                order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = 1)
                self.position = -1 # short position
                self.report_trade(order, 'GOING SHORT')
            #print(bar, self.data.index[bar].date(), self.data.SMA_S.iloc[bar], self.data.SMA_L.iloc[bar], self.data.direction.iloc[bar], sep = " | ")
        time.sleep(0.1) 

        
    def start_trading(self):      
        st.write('Start Trading') 
        self.twm = ThreadedWebsocketManager() 
        self.twm.start()
        
        if self.bar_length in self.available_intervals:
            # nice printout
            stm = "Symbol | {} | Time | {} | len(SMA_S) = {} & len(SMA_L) = {}".format(self.symbol, datetime.utcnow(),self.SMA_S, self.SMA_L)
            st.write("-" * 75)
            st.write(stm)
            st.write("-" * 75)

            # reset 
            self.trades = 0  # no trades yet
            #self.current_balance = self.initial_balance  # reset initial capital
            
            # get data
            now = datetime.utcnow()
            self.get_historical_data(symbol = self.symbol, interval = self.bar_length, start=now, end=None, most_recent_obs=self.SMA_L)
            # streaming data
            self.twm.start_kline_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length) # Adj: start_kline_futures_socket
            
            
            
    def report_testing_trade(self, going): 
        
        time.sleep(0.1)
        
        order_time = self.prepared_data.index[self.bar]
        order_time = pd.to_datetime(order_time, unit = "ms")       
        base_units = self.units
        price = self.get_values(self.bar)[1]
        quote_units = float(price * base_units)
        # total_price = round(quote_units / base_units, 5)
        print('base_units: ', base_units)
        
        # calculate trading profits
        self.trades += 1
        if self.position == 1:
            self.trade_values.append(-quote_units)
        elif self.position == -1:
        #if self.position == -1:  
            self.trade_values.append(quote_units) 
        
        if self.trades % 2 == 0:
            real_profit = round(np.sum(self.trade_values[-2:]), 3) 
            self.cum_profits = round(np.sum(self.trade_values), 3)
        else: 
            real_profit = 0
            self.cum_profits = round(np.sum(self.trade_values[:-1]), 3)
        
        # print trade report
        print(2 * "\n" + 100* "-")
        print("{} | {}".format(order_time, going)) 
        print("{} | Base_Units = {} |  Price = {} | Quote_Units = {}".format(order_time, base_units, price, quote_units))
        print("Profit = {} | CumProfits = {} ".format(real_profit, self.cum_profits))
        print(100 * "-" + "\n")
        
 
    def report_trade(self, order, going): 
        
        time.sleep(0.1)
        order_time = self.prepared_data.index[self.bar]
        order_time = pd.to_datetime(order_time, unit = "ms")       
        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)
        
        # calculate trading profits
        self.trades += 1
        side = order["side"]
        if side == "BUY":
            self.trade_values.append(-quote_units)
        elif side == "SELL":
            self.trade_values.append(quote_units) 
        
        if self.trades % 2 == 0:
            real_profit = round(np.sum(self.trade_values[-2:]), 3) 
            self.cum_profits = round(np.sum(self.trade_values), 3)
        else: 
            real_profit = 0
            self.cum_profits = round(np.sum(self.trade_values[:-1]), 3)
        
        # print trade report
        print(2 * "\n" + 100* "-")
        print("{} | {}".format(time, going)) 
        print("{} | Base_Units = {} | Quote_Units = {} | Price = {} ".format(time, base_units, quote_units, price))
        print("{} | Profit = {} | CumProfits = {} ".format(time, real_profit, self.cum_profits))
        print(100 * "-" + "\n")


    def rsi_strategy(self, rsi_period, sma_period, lower_th, upper_th):

        data = self.data.copy()
        
        data['sma'] = data['price'].rolling(sma_period).mean()
        data['rsi'] = talib.RSI(data['sma'], rsi_period)
        data['rsi_indicator'] = np.nan

        data['rsi_1'] = data['rsi'].shift(1)
        data['rsi_2'] = data['rsi'].shift(2)
        data['rsi_ratio'] = data['rsi']/data['rsi_1']

        data.loc[(data['rsi'] > upper_th), 'rsi_indicator'] = 0
        data.loc[data['rsi'] < lower_th, 'rsi_indicator'] = 1
        data.loc[(data['rsi'] > upper_th) & (data['rsi_ratio'] < 1.002), 'rsi_signal'] = 'SELL'
        data.loc[(data['rsi'] < lower_th) & (data['rsi_ratio'] > 1), 'rsi_signal'] = 'BUY'


        data['position'] = 0
        data['strategy'] = 'N/A'

        index = data.index[-1]
        
        if(data.loc[index,'rsi_signal'] == 'BUY'):
            if(self.position == 0):
                self.position = 1
                data.loc[index, 'strategy'] = 'BUY'
        elif(data.loc[index,'rsi_signal'] == 'SELL'):
            if(self.position == 1):
                self.position = 0
                data.loc[index, 'strategy'] = 'SELL'
        data.loc[index,'position'] = self.position

        return data.tail(1)    


    def rsi_back_testing(self, start_time, end_time, rsi_period, sma_period, lower_th, upper_th):
        self.get_historical_data(symbol = self.symbol, interval = "15m", start = str(start_time))

        data = self.data.copy()
        
        data['sma'] = data['price'].rolling(sma_period).mean()
        data['rsi'] = talib.RSI(data['sma'], rsi_period)
        data['rsi_indicator'] = np.nan

        data['rsi_1'] = data['rsi'].shift(1)
        data['rsi_2'] = data['rsi'].shift(2)
        data['rsi_ratio'] = data['rsi']/data['rsi_1']

        data.loc[(data['rsi'] > upper_th), 'rsi_indicator'] = 0
        data.loc[data['rsi'] < lower_th, 'rsi_indicator'] = 1
        data.loc[(data['rsi'] > upper_th) & (data['rsi_ratio'] < 1.002), 'rsi_signal'] = 'SELL'
        data.loc[(data['rsi'] < lower_th) & (data['rsi_ratio'] > 1), 'rsi_signal'] = 'BUY'


        data['position'] = 0
        data['strategy'] = np.nan
        current_position = 1

        for index, row in data.iterrows():
            if(data.loc[index,'rsi_signal'] == "BUY"):
                if(current_position == 0):
                    current_position = 1
                    data.loc[index, 'strategy'] = 'BUY'
            elif(data.loc[index,'rsi_signal'] == 'SELL'):
                if(current_position == 1):
                    current_position = 0
                    data.loc[index, 'strategy'] = 'BUY'
            data.loc[index,'position'] = current_position

        st.write(data.head())



    def rsi_trading(self, start_time, rsi_period, sma_period, lower_th, upper_th):      
        n = 0
        result_df = pd.DataFrame()
        self.get_historical_data(symbol = symbol, interval = self.bar_length, start = str(start_time))
        st.write('Latest time of 15mins interval: ', self.data.index[-1])
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        st.write('Current time: ', current_time)
        
        latest_time = self.data.index[-1]
        latest_time = pd.to_datetime(latest_time)
        next_time = latest_time + timedelta(hours = 0.25)
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        current_time = pd.to_datetime(current_time)
        # latest_time, next_time, current_time
        duration = next_time - current_time
        seconds = duration.seconds
        minutes = (seconds % 3600) // 60
        minutes = minutes
        st.write('waiting for ' , minutes)
        time.sleep(seconds)
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        current_time = pd.to_datetime(current_time)
        st.write('Current time: ', current_time)

        while True:                  
            if (current_time >= next_time):
                time.sleep(20)
                self.get_historical_data(symbol = symbol, interval = self.bar_length, start = str(start_time))
                self.data['Date'] = self.data.index
                result = self.rsi_strategy(rsi_period, sma_period, lower_th, upper_th)
                result_df = result_df.append(result)
                print(result[['price','rsi','rsi_1','rsi_ratio','rsi_signal','position','strategy']],self.symbol,'\n')
                
                if( result.strategy[0] != 'N/A'):
                    if result.strategy[0] == 'BUY': # signal to go long
                        order = client.create_order(symbol = self.symbol, side = "BUY", type = "MARKET", quantity = self.units)
                        st.write('BUY')
                    elif result.strategy[0] == 'SELL': # signal to go short
                        order = client.create_order(symbol = self.symbol, side = "SELL", type = "MARKET", quantity = self.units)
                current_time = pd.to_datetime(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
                next_time = pd.to_datetime(self.data.index[-1]) + timedelta(hours = 0.25)
            st.write('\n', str(current_time), end=" ", flush=True)
            time.sleep(60)
            #n = n + 1
            current_time = pd.to_datetime(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
            #print(str(current_time), str(next_time))



    def plot_RSI_signal(data, symbol):
        #plot result 
        fig, ax = plt.subplots(figsize=(18, 8))
        graph = sns.lineplot(data = data, x = 'Date', y = 'price', color='blue')
        sns.lineplot(data = data, x = 'Date', y = 'sma', color='red')
        graph.set_title("Bollinger Bands:" + symbol)  
        
        # RSI rsi_indicator
        rsi_indicator = data.loc[data['rsi_indicator'].isin([1,0])]
        rsi_indicator = rsi_indicator.reset_index(drop = True)
        #print(rsi_indicator.shape)
        
        # RSI signals
        rsi_signals = data.loc[data['rsi_signal'].isin(['BUY','SELL'])]
        rsi_signals = rsi_signals.reset_index(drop = True)
        
        
        #Add close price on signals 
        for x,y,z in zip(rsi_indicator['Date'], rsi_indicator['sma'], np.round(rsi_indicator['rsi'],2)):
            label = z #Label corresponds to labels in dataset
            plt.annotate(label, #text to be displayed
                        (x,y), #point for the specific label
                        textcoords="offset points", #positioning of the text
                        xytext=(0,6), #distance from text to points
                        ha='center',
                        fontsize = 13) #horizontal alignment
            
        
        #Add close price on signals 
        for x,y,z in zip(rsi_signals['Date'], rsi_signals['sma'], rsi_signals['rsi_signal']):
            label = z #Label corresponds to labels in dataset
            plt.annotate(label, #text to be displayed
                        (x,y), #point for the specific label
                        textcoords="offset points", #positioning of the text
                        xytext=(0,15), #distance from text to points
                        ha='center',
                        fontsize = 15,
                        color='red') #horizontal alignment
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)



    def plot_position(self):
        fig, ax = plt.subplots(figsize=(18, 8))
        plt.xticks(self.prepared_data.index, rotation=90)
        ax2 = ax.twinx()
        ax.plot(self.prepared_data.index, self.prepared_data[["price","SMA_S", "SMA_L"]])
        ax2.plot(self.prepared_data.index, self.prepared_data["direction"], color='red', label='Seconds')
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)


    def plot_test_performance(self):

        '''
        fig, axes = plt.subplots(1,1,figsize=(18,8))
        plt.plot(self.prepared_data[["creturns", "cstrategy"]])
        buf = BytesIO()
        fig.savefig(buf, format="png")
        st.image(buf)
        '''

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.prepared_data.index,
                y=self.prepared_data["creturns"],
                mode="lines",
                name="Buy and Hold Strategy",
                line_color="green"
            )
        )


        fig.add_trace(
            go.Scatter(
                x=self.prepared_data.index,
                y=self.prepared_data["cstrategy"],
                mode="lines",
                name="SMA Strategy",
                line_color="orange"
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

        st.write(fig)

