import streamlit as st
import datetime
import numpy as np
import plotly.graph_objects as go
from analysis.finacial_instrument import FinancialInstrument
from analysis.finacial_instrument import Spot_Trading
import config as cf


# Define parameters
DEFAULT_DAY_INTERVAL = 1000
#SYMBOL = np.array(["Select symbol","FILUSDT", "BNBUSDT", "ETCUSDT", "DOTUSDT", "XTZUSDT"])
TIME_INTERVAL = ["1d", "6h", "1h", "30m", "15m"]
TRADING_INTERVAL = ["15m", "30m", "1h", "6h", "12h"]
TRADING_STRATEGY = ["Select method", "SMA"]
TEST_INTERVAL_LENGTH = 60
TRAIN_INTERVAL_LENGTH = 1000
HORIZON = 60
TRAINED = False
TRADING_FLAG = 0



def plot_data(fi):
	#plot data
	fig = go.Figure()
	fig = fi.plot_raw_data(fig)
	fig.update_layout(
		width=st.session_state.CHART_WIDTH,
		margin=dict(l=0, r=0, t=0, b=0, pad=0),
		legend=dict(x=0, y=0.99, traceorder='normal', font=dict(size=12)),
		autosize=False,
		template="plotly_dark"

	)
	st.write(fig)	


def app():

	
	########################################## Layout setting #############################################

	# Select features
	st.sidebar.subheader('Select function')
	task_type = ['Introduction',
				 'Trading Bitcoin',
				 'Financial Data Analysis']
	task_option = st.sidebar.selectbox('', task_type)
	st.sidebar.header('')



	########################################### Time window #########################################
	window_selection_c = st.sidebar.container()  # create an empty container in the sidebar
	chart_width = st.expander(label="Select chart width").slider("",500, 2800, 1200, key="CHART_WIDTH")


	################################# Stock selection box & stock linecharts ##########################
	if(task_option == 'Trading Bitcoin'):

		# Add new symbol
		new_symbol = st.sidebar.text_input('New Symbol').capitalize()
		if st.sidebar.button('Add'):
			cf.data['bitcoin_symbol'].update(new_symbol)
			cf.update_yaml_config_file(cf.data)


		# Stock window selection
		SYMBOL = cf.data['bitcoin_symbol']
		selected_symbol = window_selection_c.selectbox("Select symbol", SYMBOL)
		time_interval = window_selection_c.selectbox("Select time interval", TIME_INTERVAL)
		if(selected_symbol != 'Select symbol'):

			# initialize class object
			fi = FinancialInstrument(symbol=selected_symbol)

			# Time window selection
			time_window_c = window_selection_c.columns(2)  # Split the container into two columns for start and end date
			DEFAULT_START = fi.get_earliest_valid_timestamp(time_interval)
			TODAY = datetime.date.today() 
			# Start date
			START = time_window_c[0].date_input("From", value=DEFAULT_START, max_value=TODAY - datetime.timedelta(days=1))
			END = time_window_c[1].date_input("To", value=TODAY, max_value=TODAY, min_value=START)

			if window_selection_c.button("Collect data"):
				fi.get_historical_bitcoin_data(time_interval, START, END, most_recent_obs=0)
				# plot raw data
				plot_data(fi)


			if window_selection_c.button("Data Exploration"):
				time_window_control = st.columns(3)
				DEFAULT_START_TRADING = TODAY - datetime.timedelta(7)
				START_TRADING = time_window_control[0].date_input("From", value=DEFAULT_START_TRADING, max_value=TODAY - datetime.timedelta(days=1))
				END_TRADING = time_window_control[1].date_input("To", value=TODAY, max_value=TODAY, min_value=START_TRADING)
				TRADING_interval = time_window_control[2].selectbox("Select time interval", TRADING_INTERVAL)
				fi = FinancialInstrument(symbol=selected_symbol)

				st.markdown('<p style="color:Green; font-size: 25px;"> 1. Plot raw data</p>', unsafe_allow_html=True)
				fi.get_historical_bitcoin_data(TRADING_interval, START_TRADING, END_TRADING, most_recent_obs=0)
				plot_data(fi)

				st.markdown('<p style="color:Green; font-size: 25px;"> 2. Simple Moving Average - SMA</p>', unsafe_allow_html=True)
				sma_window_control = st.columns(3)
				SMA_1 = sma_window_control[0].number_input('SMA_S',1, 20, 8)
				SMA_2 = sma_window_control[1].number_input('SMA_S',10, 50, 24)
				SMA_3 = sma_window_control[2].number_input('SMA_S',30, 100, 48)
				fi.explore_data(SMA_1, SMA_2, SMA_3)


			if window_selection_c.button("Backtesting"):
				time_window_control = st.columns(3)
				DEFAULT_START_TRADING = TODAY - datetime.timedelta(7)
				START_TRADING = time_window_control[0].date_input("From", value=DEFAULT_START_TRADING, max_value=TODAY - datetime.timedelta(days=1))
				END_TRADING = time_window_control[1].date_input("To", value=TODAY, max_value=TODAY, min_value=START_TRADING)
				TRADING_interval = time_window_control[2].selectbox("Select time interval", TRADING_INTERVAL)

				rsi_params = st.columns(4)
				rsi_period = rsi_params[0].number_input('rsi_period',1, 20, 4)
				sma_period = rsi_params[1].number_input('sma_period',1, 20, 8)
				lower_th = rsi_params[2].number_input('lower_th',1, 100, 30)
				upper_th = rsi_params[3].number_input('upper_th',1, 100, 70)

				trader = Spot_Trading(selected_symbol, TRADING_interval, units=1, position=0)
				trader.rsi_back_testing(START_TRADING, END_TRADING ,rsi_period, sma_period, lower_th, upper_th)







		'''	
		# selected_method = window_selection_c.selectbox("Select strategy", TRADING_STRATEGY)
		SMA_S = st.sidebar.number_input('SMA_S',10, 60, 20)
		SMA_L = st.sidebar.number_input('SMA_L',30, 100, 48)
		units = 1
		position = 0
		if window_selection_c.button("SMA Strategy"):
			trader = Spot_SMA(symbol = selected_symbol, bar_length = time_interval,
                       sma_s = SMA_S, sma_l = SMA_L, 
                       units = units, position = position)

			trader.get_historical_data(selected_symbol, time_interval, str(START), str(END), SMA_L)
			trader.SMA_strategy()
			st.write(trader.prepared_data.tail())
			st.write(trader.testing_SMA([SMA_S, SMA_L]))
			
			trader.plot_test_performance()
			trader.plot_SMA()

		if window_selection_c.button("Start Trading"):
			trader = Spot_SMA(symbol = selected_symbol, bar_length = time_interval,
                       sma_s = SMA_S, sma_l = SMA_L, 
                       units = units, position = position)
			trader.start_trading()
		'''















	
