"""
This file is the framework for generating multiple Streamlit applications 
through an object oriented framework
"""

# Import libraries
import streamlit as st

# Define the multiplage class to manage the multiple apps 
class MultiPage:

	def __init__(self, col) -> None:
		# Constructor class to generate a list which stores all applications as an instance variable
		self.pages = []
		self.col = col

	def add_page(self, title, func) -> None:
		""" Class Method to add pages to the project

		Args:
			title ([str]): the title of page which we are adding to the list of apps
			func: Python function to render this page in Streamlit

		"""

		self.pages.append(
			{
				"title": title,
				"function": func
			}
		)

	def run(self):

		# Add dropdown to select the page to run
		#st.sidebar.header('Applications')
		page = st.sidebar.selectbox(
			'',
			self.pages,
			format_func=lambda page: page['title'])

		# run the app function
		if(page['title'] == 'Select Application'):
			title = '<p style="font-family:sans-serif; color:Pink; font-size: 40px;"> Introduction</p>'
		if(page['title'] == '01-House Price Data Analysis'):
			title = '<p style="font-family:sans-serif; color:Pink; font-size: 40px;"> Regression Analysis</p>'
		if(page['title'] == '02-Credit Risk Analysis'):
			title = '<p style="font-family:sans-serif; color:Pink; font-size: 40px;"> Classification Analysis</p>'
		if(page['title'] == '03-Unsupervised Techniques'):
			title = '<p style="font-family:sans-serif; color:Pink; font-size: 40px;"> Unsupervised Analysis</p>'
		if(page['title'] == '04-Stock Forecast Dashboard'):
			title = '<p style="font-family:sans-serif; color:Pink; font-size: 40px;"> Time Series Analysis</p>'
		if(page['title'] == '05-Hotel Recommendation'):
			title = '<p style="font-family:sans-serif; color:Pink; font-size: 40px;"> Natural Language Processing</p>'
		if(page['title'] == '06-Online Shopping Recommendation'):
			title = '<p style="font-family:sans-serif; color:Pink; font-size: 40px;"> Recommender System</p>'


		self.col.markdown(title, unsafe_allow_html=True)
		page['function']()

