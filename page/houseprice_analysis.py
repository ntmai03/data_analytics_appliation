import streamlit as st


def app():
	st.sidebar.subheader('Select function')
	task_type = ['Introduction',
				 'Exploratory Data Analysis',
				 'Data Processing',
				 'Predictive Model',
				 'Prediction']
	task_option = st.sidebar.selectbox('', task_type)
	st.sidebar.header('')

	if task_option == 'Introduction':
		st.write('Comming soon...')