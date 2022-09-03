import imp
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("ML MODEL TO PREDICT MALACIOUS URLs")


st.header("header")
st.subheader("Sub header")
st.text("o;aiu;foaf;af;a")


st.markdown(""" # h1 tag
## h2 tag
### h3 tag
:moon:<br>
:sunglasses:
 """, True)


a = [1,2,3,4,5,6,7,8]
n = np.array(a)
nd = n.reshape((2,4))

dic = {
    "name": "Priyanshu",
    "age": 21,
    "city": "Lucknow"

}


data = pd.read_csv('E:\Project_2\Machine-Learning-for-Security-Analysts-master\Malicious URLs.csv')


st.dataframe(data)