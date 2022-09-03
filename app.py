from pyexpat import model
from django import urls
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

from mal import tVec
from mal import model



# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
model = pickle.load(pickle_in)



def welcome():
	return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs

# x_predict= tVec.transform(x_predict)
# New_predict = mnb_count.predict(x_predict) 
def prediction(x_predict):

	prediction = model.predict(
		[[x_predict]])
	print(prediction)
	return prediction
	

# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	st.title("Malicious URLs Prediction")
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Streamlit Malicious URLs  ML App </h1>
	</div>
	"""
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)
	
	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
    x_predict= tVec.fit_transform(x_predict)
    print(x_predict.type())
	x_predict = st.text_input("URLS", "Type Here")

	result =""
	
	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
	if st.button("Predict"):
		result = prediction(x_predict)
	st.success('The output is {}'.format(result))
	
if __name__=='__main__':
	main()
