import streamlit as st
import datetime
import numpy as np
import plotly.graph_objects as go
from analysis.stock import Stock


# Define parameters
DEFAULT_DAY_INTERVAL = 700
STOCKS = np.array(["GOOG", "GME", "FB", "APL", "TSLA"])
TEST_INTERVAL_LENGTH = 60
TRAIN_INTERVAL_LENGTH = 500
HORIZON = 60
TRAINED = False


def app():

	
	########################################## Layout setting #############################################

	#------------------------------ Time window and Stock selection box------------------------------
	window_selection_c = st.sidebar.container()  # create an empty container in the sidebar
	window_selection_c.markdown("## Filter")  # add a title to the sidebar container

	# Time window selection
	time_window_c = window_selection_c.columns(2)  # Split the container into two columns for start and end date
	# set default day as YESTERDAY
	YESTERDAY = datetime.date.today() - datetime.timedelta(days=1)
	# get the nearest business day
	YESTERDAY = Stock.nearest_business_day(YESTERDAY)
	DEFAULT_START = YESTERDAY - datetime.timedelta(days=DEFAULT_DAY_INTERVAL)
	DEFAULT_START = Stock.nearest_business_day(DEFAULT_START)
	# Start date
	START = time_window_c[0].date_input("From", value=DEFAULT_START, max_value=YESTERDAY - datetime.timedelta(days=1))
	START = Stock.nearest_business_day(START) 
	END = time_window_c[1].date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)
	END = Stock.nearest_business_day(END)

	# Stock window selection
	selected_stock = window_selection_c.selectbox("Select Stock", STOCKS)
	chart_width = st.expander(label="Select chart width").slider("",500, 2800, 1200, key="CHART_WIDTH")


	########################################## Plot stock linecharts #########################################
	# initialize class object
	stock = Stock(symbol=selected_stock)
	stock.load_data(START, END, inplace=True)

	#plot data
	fig = go.Figure()
	fig = stock.plot_raw_data(fig)
	fig.update_layout(
		width=st.session_state.CHART_WIDTH,
		margin=dict(l=0, r=0, t=0, b=0, pad=0),
		legend=dict(x=0, y=0.99, traceorder='normal', font=dict(size=12)),
		autosize=False,
		template="plotly_dark"

	)
	st.write(fig)

	# display current selected stock price
	with window_selection_c:
		stock.show_delta()

	


	########################################## FORECAST #############################################

	#------------------------------------ Forecast controls box-----------------------------------
	forecast_c = st.sidebar.container()
	forecast_c.markdown("## Forecasts")

	# Session state initialization
	if "TEST_INTERVAL_LENGTH" not in st.session_state:
		# set the initial default value of test interval
		st.session_state.TEST_INTERVAL_LENGTH = 60

	if "TRAIN_INTERVAL_LENGTH" not in st.session_state:
		# set the initial default value of the training lenght widget
		st.session_state.TRAIN_INTERVAL_LENGTH = 500

	if "TRAIN" not in st.session_state:
		st.session_state.TRAINED = False

	# Train time window selection
	train_window_c = forecast_c.columns(2)
	time_interval = (END - START).days
	train_days = int(3*time_interval/4)
	TRAIN_END =  train_window_c[1].date_input("Train_To", value= START + datetime.timedelta(train_days), max_value=END - datetime.timedelta(days=1))
	TRAIN_END = Stock.nearest_business_day(TRAIN_END) 
	TRAIN_START = train_window_c[0].date_input("Train_From", value=START, max_value=TRAIN_END - datetime.timedelta(days=1))
	TRAIN_START = Stock.nearest_business_day(TRAIN_START) 
	st.session_state.TRAIN_START = TRAIN_START
	st.session_state.TRAIN_END = TRAIN_END

	test_days = int(time_interval/4)
	TEST_START = TRAIN_END + datetime.timedelta(days=1)
	TEST_START = Stock.nearest_business_day(TEST_START) 
	TEST_END = END
	TEST_END = Stock.nearest_business_day(TEST_END) 
	st.session_state.TEST_START = TEST_START
	st.session_state.TEST_END = TEST_END

	train_test_split = forecast_c.columns(2)
	train_test_split[0].markdown(" \# train days: " + str(train_days))
	train_test_split[1].markdown(" \# test days: " + str(test_days))

	forecast_c.markdown("Start Test " + str(TEST_START))
	forecast_c.markdown("End Test " + str(TEST_END))

	forecast_c.button(
		label="Train",
		key='TRAIN_JOB'
	)


	st.write("-----------------------------------")
	st.markdown("### Train Model")
	stock.train_test_forecast_report(selected_stock)

	# Reference
    # https://python.plainenglish.io/building-a-stock-market-app-with-python-streamlit-in-20-minutes-2765467870ee'




