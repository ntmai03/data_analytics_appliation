import streamlit as st
import datetime
import time
from datetime import timedelta
import pandas as pd
import numpy as np

from PIL import Image
from io import BytesIO

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.patheffects as PathEffects

# defined functions and libraries
from analysis.coin_trading import Coin_Trading
import config as cf
import util.timeseries_util as tsu
import util.data_manager as dm

def app():

	#============================================= Display Menu ========================================
	st.sidebar.subheader('Select function')
	task_type = ['Introduction',
				 'Add new symbol',
				 'Data Collection',
				 'Trading Analysis',
				 'Trading Optimization',
				 'Trading Automation',
				 'Trading Monitoring']
	task_option = st.sidebar.selectbox('', task_type)
	st.sidebar.header('')


	#======================================== PART I: INTRODUCTION ======================================
	if (task_option == 'Introduction'):
		st.markdown('<p style="color:lightgreen; font-size: 30px;"> 1. Overview</p>', unsafe_allow_html=True)
		st.write("**Introduction to Binance API**: Binance API provides API for retriving information about coins such as market price, volume, and we can also stream data, place orders and makes trade with Python ")
		st.write('Coin Trading is an End2End application which allows to get historical crypto price from Binance platform, perform analyzing data and automatically trade crypto with the following tasks:')
		st.write("**1. Add new symbol** : This function allows to add a new symbol defined in Binance platform to display on symbol list")
		st.write("**2. Data Collection**: This function calls Binance API to retrieve historical data of a certain symbol/bitcoin")
		st.write("**3. Data Exploration**: Apply some visualization techniques and technical indicators to examine data")
		st.write("**4. Trading Strategy**:  This function allows to perform back testing differnt trading strategies using SMA and RSI indicators and analyzing performance of the strategy ")
		st.write("**5. Trading Demo**: The application demos automatically trading every 15 minutes. Due to sensitive info, it doesn't execute real buying or selling orders")

		st.markdown('<p style="color:lightgreen; font-size: 30px;"> 2. Demo</p>', unsafe_allow_html=True)
		video_file = open('trading.mp4', 'rb')
		video_bytes = video_file.read()
		st.video(video_bytes)

	#===================================== PART II:  ADD AND DISPLAY COIN LIST ===================================
	if (task_option == 'Add new symbol'):
		st.write("Input a new symbol that exists on Binance and click Add button to add this symbol to the symbol list")
		new_symbol = st.sidebar.text_input('Input new symbol').upper()
		if st.sidebar.button('Add'):
			cf.data['BITCOIN_SYMBOL'].append(new_symbol)
			cf.update_yaml_config_file(cf.data)

	# display list of coins, list of coins is stored in file config.yml
	SYMBOL = cf.data['BITCOIN_SYMBOL']
	selected_symbol = st.sidebar.selectbox("Select symbol", SYMBOL)



	#===================================== PART III: DATA COLLECTION ===================================
	if (task_option == 'Data Collection'):
		st.write('By default, the function get earliest available data of a selected symbol on Binance')
		if(selected_symbol != 'Select symbol'):	
			# Split the container into two columns for start, end date and time interval
			ct = Coin_Trading(symbol=selected_symbol)
			
			# get earliest available data on Binance
			DEFAULT_START = ct.get_earliest_valid_timestamp('1d')
			time_window_cotrol = st.sidebar.columns(3)  
			TODAY = datetime.date.today() 
			START = time_window_cotrol[0].date_input("From", value=DEFAULT_START, max_value=TODAY - datetime.timedelta(days=1), min_value=DEFAULT_START)
			END = time_window_cotrol[1].date_input("To", value=TODAY, min_value=START)	
			time_interval = time_window_cotrol[2].selectbox("Select time interval", cf.data['BITCOIN_TIME_INTERVAL'])
			
			if(st.sidebar.button('Collect data')):
				ct.get_historical_bitcoin_data(time_interval, START, END, most_recent_obs=0)
				tsu.plot_timeserie_data(ct.data.Date, ct.data.price, selected_symbol)



	#===================================== PART IV: DATA EXPLORATION ===================================
	if (task_option == 'Trading Analysis'):
		time_window_control = st.sidebar.columns(3)
		TODAY = datetime.date.today() 
		DEFAULT_START_TRADING = TODAY - datetime.timedelta(cf.data['default_start_trading'])
		START_TRADING = time_window_control[0].date_input("From", value=DEFAULT_START_TRADING, max_value=TODAY - datetime.timedelta(days=1))
		END_TRADING = time_window_control[1].date_input("To", value=TODAY, min_value=START_TRADING)
		trading_interval = time_window_control[2].selectbox("Select time interval", cf.data['BITCOIN_TRADING_INTERVAL'])		
		rsi_period = st.sidebar.number_input('rsi_period',1, 100, 20)
		sma_period = st.sidebar.number_input('sma_period',1, 100, 20)
		lower_threshold = st.sidebar.number_input('lower_threshold',1, 100, 10)
		upper_threshold = st.sidebar.number_input('upper_threshold',1, 100, 90)
		cutloss_flag = st.sidebar.number_input('cutloss_flag',0, 1, 1)
		increase_flag = st.sidebar.number_input('increase_flag',0, 1, 1)
		rsi_limit1 = 1
		rsi_limit2 = 1
		cutloss_th=2

		if (selected_symbol != 'Select symbol'):
			if(st.sidebar.button('Plot data')):
				st.markdown('<p style="color:lightgreen; font-size: 30px;"> Examine the best SMA trading intervals</p>', unsafe_allow_html=True)
				st.write("SMA (simple moving average) is a technical indicator that 'smooth out' fluctuations by averaging a fixed number of data points to help distinguish between typical market flutuations and actual movement")
				st.write("Firstly, we need to specify number of data points for calcullating rolling average. In order to choose the best time interval for trading, we can try with different values. The plot below using 3 SMAs (SMA_1, SMA_2, SMA_3) along the actual price in order to examine and compare which length value matches the price movement best to get as much  trading opportunities as possible")
				SMA_1 = st.sidebar.number_input('SMA_1',1, 20, 6)
				SMA_2 = st.sidebar.number_input('SMA_2',10, 30, 10)
				SMA_3 = st.sidebar.number_input('SMA_3',20, 50, 20)
				ct = Coin_Trading(symbol=selected_symbol)
				# plot raw data
				st.markdown('<p style="color:lightgreen; font-size: 25px;"> 1. Plot raw data</p>', unsafe_allow_html=True)
				ct.get_historical_bitcoin_data(trading_interval, START_TRADING, END_TRADING, most_recent_obs=0)
				tsu.plot_timeserie_data(ct.data.Date, ct.data.price, selected_symbol)
				st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. Simple Moving Average - SMA</p>', unsafe_allow_html=True)
				ct.explore_data(SMA_1, SMA_2, SMA_3)
				st.write("It can be seen from the plot that if we follow sma=6 (red line), there are more opportunities to trade because it recognizes the pattern (increasing trend, decreasing trend) soon and make better decisions.")

				st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. RSI Introduction</p>', unsafe_allow_html=True)
				st.write("RSI (Relative Strength Index) is a type of momentum oscillator which fluctuates between 0 and 100. Typically, if the RSI goes above 70 this indicates that the stock is overpriced whilst if it goes below 30 the stock is oversold. This helps traders or investors measure the speed and change of a share price")
				st.markdown('<p style="color:lightgreen; font-size: 25px;"> 2. The solution covers 3 following cases</p>', unsafe_allow_html=True)
				st.write("Case 1 - using only rsi and rsi_ratio to estimate the trend")
				st.write("Case 2 - cut loss in case price going down for a long time (bear market): select cutloss_flag = 1")
				st.write("Case 3 - cut loss in bear market or enter in bull market: select cutloss_flag = 1 and increase_flag = 1")
				st.markdown('<p style="color:lightgreen; font-size: 25px;"> 3. for each case, there are 3 scenarios tested for symbol WAVEUSDT</p>', unsafe_allow_html=True)
				st.write("Scenario 1 - price fluctuates: from 2022-02-10 to 2022-02-20")
				st.write("Scenario 2 - price increasing: from 2022-03-20 to 2022-03-30")
				st.write("Scenario 3 - price decreasing: from 2022-04-01 to 2022-04-10")


		if(st.sidebar.button('Investigating')):
			st.write("Please note that if time range is long, it only displays charts of the first 10 days for investigating")
			ct = Coin_Trading(symbol=selected_symbol)
			ct.get_historical_bitcoin_data(trading_interval, START_TRADING, END_TRADING, most_recent_obs=0)
			ct.rsi_back_testing(rsi_period, sma_period, 
								lower_threshold, upper_threshold, 
								rsi_limit1=rsi_limit1,rsi_limit2=rsi_limit2,
								cutloss_flag=cutloss_flag, cutloss_th=cutloss_th, increase_flag=increase_flag)
			tsu.plot_timeserie_data(ct.trading_data.Date, ct.trading_data.price, selected_symbol)
			start_date = pd.to_datetime(START_TRADING)
			end_date = pd.to_datetime(END_TRADING)
			n_days = end_date - start_date
			if (n_days.days > 20):
				n_days = 20
			else:
				n_days = n_days.days

			for i in range(0, n_days, 1):
				start_index = 96*i + 0
				end_index = 96*i + 96
				current_date = ct.trading_data.iloc[start_index : end_index]
				ct.plot_RSI_signal(ct.trading_data.iloc[start_index:end_index,:], selected_symbol)
			ct.calculate_performance()
			ct.plot_test_performance()
			ct.plot_position()	
		


	#===================================== PART V: TRADING STRATEGY WITH RSI ===================================
	if (task_option == 'Trading Optimization'):
		if (selected_symbol != 'Select symbol'):
			time_window_control = st.sidebar.columns(3)
			TODAY = datetime.date.today() 
			DEFAULT_START_TRADING = TODAY - datetime.timedelta(cf.data['default_start_trading'])
			START_TRADING = time_window_control[0].date_input("From", value=DEFAULT_START_TRADING, max_value=TODAY - datetime.timedelta(days=1))
			END_TRADING = time_window_control[1].date_input("To", value=TODAY, min_value=START_TRADING)
			trading_interval = time_window_control[2].selectbox("Select time interval", cf.data['BITCOIN_TRADING_INTERVAL'])		
			lower_threshold = st.sidebar.number_input('lower_threshold',1, 100, 10)
			upper_threshold = st.sidebar.number_input('upper_threshold',1, 100, 90)
			cutloss_flag = st.sidebar.number_input('cutloss_flag',0, 1, 1)
			increase_flag = st.sidebar.number_input('increase_flag',0, 1, 1)
			rsi_limit1 = 1
			rsi_limit2 = 1
			cutloss_th=2

			if(st.sidebar.button('Find parameter')):
				ct = Coin_Trading(symbol=selected_symbol)
				ct.get_historical_bitcoin_data(trading_interval, START_TRADING, END_TRADING, most_recent_obs=0)
				rsi_creturns = []
				for time_step in range(6, 100):
				    st.write(time_step)
				    rsi_period = time_step
				    sma_period = time_step
				    ct.rsi_back_testing(rsi_period, sma_period, lower_threshold, upper_threshold, rsi_limit1=rsi_limit1, rsi_limit2=rsi_limit2,
				                  cutloss_flag=cutloss_flag, cutloss_th=cutloss_th, increase_flag=increase_flag)    
				    ct.calculate_performance()
				    st.write(ct.trading_data.iloc[-1].creturns, ct.trading_data.iloc[-1].strategy_creturns)
				    rsi_creturns.append(ct.trading_data.iloc[-1].strategy_creturns)
				st.write(np.array(rsi_creturns).max(),  np.array(rsi_creturns).argmax() + 6)
				fig, axes = plt.subplots(1,1,figsize=(12,5))
				#rsi_creturns.plot()
				plt.plot(range(6,100) ,rsi_creturns)
				buf = BytesIO()
				fig.savefig(buf, format="png")
				st.image(buf)


	if (task_option == 'Trading Automation'):
		st.write("The demo only allows to try with DOTUSDT or LINKUSDT")
		if((selected_symbol == 'DOTUSDT') | (selected_symbol == 'LINKUSDT')):
			TODAY = datetime.date.today() 
			start_time = TODAY - datetime.timedelta(cf.data['default_start_trading'])
			rsi_period = st.sidebar.number_input('rsi_period',1, 100, 20)
			sma_period = st.sidebar.number_input('sma_period',1, 100, 20)
			lower_threshold = st.sidebar.number_input('lower_threshold',1, 100, 10)
			upper_threshold = st.sidebar.number_input('upper_threshold',1, 100, 90)
			cutloss_flag = st.sidebar.number_input('cutloss_flag',0, 1, 1)
			increase_flag = st.sidebar.number_input('increase_flag',0, 1, 1)
			position = st.sidebar.number_input('position',0, 0, 1)
			units = st.sidebar.number_input('units',0, 2, 2)

			bar_length = "15m"
			rsi_limit1 = 1
			rsi_limit2 = 1
			cutloss_th=2

			if(st.sidebar.button('Start Trading')):
				ct = Coin_Trading(symbol=selected_symbol)
				#ct.rsi_trading(start_time, rsi_period, sma_period, lower_th, upper_th, bar_length, units, position)
				ct.rsi_trading(start_time, bar_length, rsi_period, sma_period, lower_threshold, upper_threshold, 
					rsi_limit1=rsi_limit1,rsi_limit2=rsi_limit2,position = position, cutloss_flag=1, cutloss_th=cutloss_th, increase_flag=1, units=units)


	if (task_option == 'Trading Monitoring'):
		if (selected_symbol != 'Select symbol'):
			ct = Coin_Trading(symbol=selected_symbol)
			file_name="/".join([cf.S3_DATA_CRYPTO_PATH, selected_symbol + '_trading.csv'])
			trading_df = dm.read_csv_file(bucket_name=cf.S3_DATA_PATH, file_name=file_name, type='s3')
			# trading_df.index = trading_df['Date']
			trading_column = ['price', 'Volume', 'rsi', 'rsi_1', 'rsi_ratio', 'position', 'rsi_signal', 'strategy']
			st.write(trading_df)
			start_date =  pd.to_datetime(trading_df.index[0])
			end_date =  pd.to_datetime(trading_df.index[-1]) + timedelta(hours = 24)
			n_days = end_date - start_date
			for i in range(0, n_days.days, 1):
				start_index = 96*i + 0
				end_index = 96*i + 96
				current_date = trading_df.iloc[start_index : end_index]
				ct.plot_RSI_signal(trading_df.iloc[start_index:end_index,:], selected_symbol)

			






	
