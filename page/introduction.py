import streamlit as st


def app():
	introduction = '<p style="font-family:sans-serif; color:Pink; font-size: 30px;">Introduction</p>'
	st.markdown(introduction, unsafe_allow_html=True)
	st.write('01-House Price Analysis')