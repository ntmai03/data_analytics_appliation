"""
This file is the framework for generating multiple Streamlit applications 
through an object oriented framework
"""

# Import libraries
import streamlit as st

# Define the multiplage class to manage the multiple apps 
class MultiPage:

	def __init__(self) -> None:
		# Constructor class to generate a list which stores all applications as an instance variable
		self.pages = []

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
		st.sidebar.header('Applications')
		page = st.sidebar.selectbox(
			'',
			self.pages,
			format_func=lambda page: page['title'])

		# run the app function
		page['function']()

