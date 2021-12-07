import streamlit as st
import datetime
import numpy as np
import plotly.graph_objects as go
from pipeline.stock import Stock


def app():
	#------------------------------- Layout setting--------------------------------
	window_selection_c = st.sidebar.container()  # create an empty container in the sidebar
	window_selection_c.markdown("## Filter")  # add a title to the sidebar container
	sub_columns = window_selection_c.columns(2) # Split the container into two columns for start and end date


	#------------------------------ Time windonw selection--------------------------
	YESTERDAY = datetime.date.today() - datetime.timedelta(days=1)
	YESTERDAY = Stock.nearest_business_day(YESTERDAY)  # Round to business day
	DEFAULT_START = YESTERDAY - datetime.timedelta(days=700)
	DEFAULT_START = Stock.nearest_business_day(DEFAULT_START)

	START = sub_columns[0].date_input("From", value=DEFAULT_START, max_value=YESTERDAY - datetime.timedelta(days=1))
	END = sub_columns[1].date_input("To", value=YESTERDAY, max_value=YESTERDAY, min_value=START)
	START = Stock.nearest_business_day(START)
	END = Stock.nearest_business_day(END)

	
	#---------------------------------Stock Slection----------------------------------
	STOCKS = np.array(["GOOG", "GME", "FB", "APPL", "TSLA"])
	SYMB = window_selection_c.selectbox("Select Stock", STOCKS)

	chart_width = st.expander(label="Select chart width").slider("", 500, 2800, 1200, key='CHART_WIDTH')


	#---------------------------------Plot stock linecharts----------------------------
	fig = go.Figure()
	stock = Stock(symbol=SYMB)
	stock.load_data(START, END, inplace=True)
	
	fig = stock.plot_raw_data(fig)
	fig.update_layout(
		width=st.session_state.CHART_WIDTH,
		margin=dict(l=0, r=0, t=0, b=0, pad=0),
		legend=dict(x=0, y=0.99, traceorder="normal", font=dict(size=12)),
		autosize=False,
		template="plotly_dark"
	)
	st.write(fig)

	change_c = st.sidebar.container()
	with change_c:
		stock.show_delta()



	#----------------------------------Session state initialization---------------------
	if "TEST_INTERVAL_LENGTH" not in st.session_state:
		# set the initial default value of test interval
		st.session_state.TEST_INTERVAL_LENGTH = 60

	if "TRAIN_INTERVAL_LENGTH" not in st.session_state:
		# set the initial default value of the training lenght widget
		st.session_state.TRAIN_INTERVAL_LENGTH = 500

	if "HORIZON" not in st.session_state:
		st.session_state.HORIZON = 60

	if "TRAIN" not in st.session_state:
		st.session_state.TRAINED = False


	#----------------------------------Session state initialization---------------------

	st.sidebar.markdown("## Forecasts")
	train_test_forecast_c = st.sidebar.container()

	train_test_forecast_c.markdown("#### Select interval lengths")

	HORIZON = train_test_forecast_c.number_input(
		"Inference horizon", min_value=7, max_value=200, key="HORIZON"
	)

	TEST_INTERVAL_LENGTH = train_test_forecast_c.number_input(
		"number of days to test on and visualize",
		min_value=7,
		key="TEST_INTERVAL_LENGTH",
	)

	TRAIN_INTERVAL_LENGTH = train_test_forecast_c.number_input(
		"number of  day to use for training",
		min_value=60,
		key="TRAIN_INTERVAL_LENGTH",
	)

	train_test_forecast_c.button(
		label="Train",
		key='TRAIN_JOB'
	)

	#Stock.train_test_forecast_report(SYMB)
    # st.write('https://python.plainenglish.io/building-a-stock-market-app-with-python-streamlit-in-20-minutes-2765467870ee')




